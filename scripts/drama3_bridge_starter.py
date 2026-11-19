#!/usr/bin/env python3
"""Minimal Drama-3 HTTP Speech Bridge & Streaming Resampler (44.1 kHz -> 24 kHz).

Wraps a loopback TensorFold Drama-3 server (`/v1/audio/speech`) with:
  - GET  /healthz            -> upstream readiness probe
  - POST /warmup             -> pre-warm up to 4 LRU VoiceCache slots
  - POST /v1/audio/speech    -> pass-through or 24 kHz WAV / raw PCM streaming
  - Pluggable text normalization via `hooks/prompt_builder.py`

Bind to 127.0.0.1 by default; place behind an authenticated proxy for LAN access.
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hooks.prompt_builder import build_tts_text  # noqa: E402

DRAMA3_HOST = os.getenv("DRAMA3_HOST", "127.0.0.1")
DRAMA3_PORT = int(os.getenv("DRAMA3_PORT", "8013"))
VOICES_DIR = Path(os.getenv("DRAMA3_VOICES_DIR", str(REPO_ROOT / "voices")))
UPSTREAM_BASE = f"http://{DRAMA3_HOST}:{DRAMA3_PORT}"

SRC_SR = 44100
TARGET_SR = 24000

app = FastAPI(title="Drama-3 Starter Speech Bridge", version="1.0.0")


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=1200)
    voice: str = Field(default="default")
    response_format: str = Field(default="wav", pattern="^(wav|pcm24k)$")
    stream: bool = Field(default=False)
    reference_audio_path: str | None = Field(default=None)
    reference_text: str | None = Field(default=None)


def _load_voice_reference(voice: str, explicit_path: str | None, explicit_text: str | None) -> tuple[str | None, str | None]:
    wav_path = Path(explicit_path) if explicit_path else (VOICES_DIR / f"{voice}.wav")
    txt_path = wav_path.with_suffix(".txt")
    if not wav_path.exists():
        return None, None
    audio_b64 = base64.b64encode(wav_path.read_bytes()).decode("ascii")
    ref_text = explicit_text
    if ref_text is None and txt_path.exists():
        ref_text = txt_path.read_text(encoding="utf-8").strip()
    return audio_b64, ref_text


def _resample_wav_to_24k_pcm(wav_bytes: bytes) -> bytes:
    """Convert 44.1 kHz mono s16le WAV to 24 kHz raw s16le PCM using torchaudio or scipy/numpy."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    if sr == TARGET_SR and nch == 1:
        return raw

    import numpy as np

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        samples = samples.reshape(-1, nch).mean(axis=1)
    try:
        import torch
        import torchaudio.functional as AF

        t = torch.from_numpy(samples).unsqueeze(0)
        out = AF.resample(t, sr, TARGET_SR).squeeze(0).clamp(-1.0, 1.0).numpy()
    except ImportError:
        n_out = max(1, int(round(len(samples) * TARGET_SR / sr)))
        x_old = np.linspace(0.0, 1.0, len(samples), endpoint=False)
        x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
        out = np.interp(x_new, x_old, samples)
    return (np.clip(out, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def _call_upstream_speech(payload: dict) -> bytes:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{UPSTREAM_BASE}/v1/audio/speech",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise HTTPException(status_code=exc.code, detail=f"Upstream Drama-3 error: {detail}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upstream Drama-3 unreachable: {exc}") from exc


@app.get("/healthz")
def healthz() -> JSONResponse:
    try:
        with urllib.request.urlopen(f"{UPSTREAM_BASE}/health", timeout=3) as resp:
            ok = resp.status == 200
    except Exception:
        ok = False
    return JSONResponse({"status": "ok" if ok else "degraded", "upstream": UPSTREAM_BASE})


@app.post("/warmup")
def warmup(voices: list[str] | None = None) -> JSONResponse:
    """Warm up to 4 reference voices into the TensorFold LRU VoiceCachePool."""
    targets = voices or sorted(p.stem for p in VOICES_DIR.glob("*.wav"))[:4]
    warmed: list[str] = []
    for name in targets[:4]:
        ref_b64, ref_text = _load_voice_reference(name, None, None)
        if not ref_b64:
            continue
        payload: dict = {
            "input": "Hello.",
            "response_format": "wav",
            "reference_audio": ref_b64,
        }
        if ref_text:
            payload["reference_text"] = ref_text
        _call_upstream_speech(payload)
        warmed.append(name)
    return JSONResponse({"warmed_voices": warmed, "lru_slots": 4})


@app.post("/v1/audio/speech")
def synthesize(req: SpeechRequest):
    text = build_tts_text(req.input)
    payload: dict = {
        "input": text,
        "response_format": "wav",
    }
    ref_b64, ref_text = _load_voice_reference(req.voice, req.reference_audio_path, req.reference_text)
    if ref_b64:
        payload["reference_audio"] = ref_b64
        if ref_text:
            payload["reference_text"] = ref_text

    wav_bytes = _call_upstream_speech(payload)
    if req.response_format == "wav":
        return Response(content=wav_bytes, media_type="audio/wav")

    pcm = _resample_wav_to_24k_pcm(wav_bytes)
    if not req.stream:
        return Response(content=pcm, media_type="application/octet-stream")

    def _iter_chunks(buf: bytes, chunk_size: int = 4800) -> Iterator[bytes]:
        for i in range(0, len(buf), chunk_size):
            yield buf[i : i + chunk_size]

    return StreamingResponse(_iter_chunks(pcm), media_type="application/octet-stream")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("DRAMA3_BRIDGE_HOST", "127.0.0.1")
    port = int(os.getenv("DRAMA3_BRIDGE_PORT", "8014"))
    uvicorn.run(app, host=host, port=port)
