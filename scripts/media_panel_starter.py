#!/usr/bin/env python3
"""Drama-3 -> MMH3 Unified Audio-Driven Video Starter Pipeline.

End-to-end single-pass pipeline:
  1. Synthesize clean dialogue WAV via local Drama-3 (`/v1/audio/speech`).
  2. Probe exact audio duration so video length aligns tightly with speech.
  3. Build the visual prompt via `hooks/prompt_builder.py` + `presets/starter_presets.yaml`.
  4. Run `mmh3 generate` with `--reference-audio` + `--lock-audio` (from the included MIT patch)
     and optional `--reference` portrait image (`<Picture 1>`).
  5. Finalize output via `hooks/post_process.py` (clean mux & metadata stripping).

Swap `hooks/prompt_builder.py`, `hooks/post_process.py`, and `presets/` locally
to plug in your own custom prompt compilers, grading, or multi-take QA workflows.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import subprocess
import sys
import urllib.request
import wave
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hooks.post_process import run_post_process  # noqa: E402
from hooks.prompt_builder import build_tts_text, build_video_prompt  # noqa: E402


def load_presets(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data.get("presets", {}) if isinstance(data, dict) else {}
    except ImportError:
        return {
            "dialogue_closeup": {
                "mmh3": {"width": 768, "height": 1280, "fps": 24, "steps": 24, "precision": "nvfp4", "lock_audio": True},
                "prompt_prefix": "Cinematic medium close-up portrait, natural studio key light.",
                "prompt_suffix": "Smooth natural lip sync matching the reference dialogue.",
            }
        }


def synthesize_drama3(
    endpoint: str,
    text: str,
    out_wav: Path,
    ref_audio: Path | None = None,
    ref_text: str | None = None,
) -> float:
    payload: dict[str, Any] = {
        "input": text,
        "response_format": "wav",
    }
    if ref_audio is not None:
        payload["reference_audio"] = base64.b64encode(ref_audio.read_bytes()).decode("ascii")
        if ref_text:
            payload["reference_text"] = ref_text

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        wav_bytes = resp.read()

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    out_wav.write_bytes(wav_bytes)
    with wave.open(str(out_wav), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def main() -> int:
    parser = argparse.ArgumentParser(description="Drama-3 -> MMH3 audio-driven starter pipeline")
    parser.add_argument("--dialogue", required=True, help="Spoken line for Drama-3 TTS")
    parser.add_argument("--scene-prompt", default="", help="Visual scene description for MMH3")
    parser.add_argument("--preset", default="dialogue_closeup", help="Preset name in presets/starter_presets.yaml")
    parser.add_argument("--presets-file", type=Path, default=REPO_ROOT / "presets/starter_presets.yaml")
    parser.add_argument("--ref-voice", type=Path, default=None, help="Optional reference voice WAV for Drama-3")
    parser.add_argument("--ref-voice-text", default=None, help="Transcript matching --ref-voice")
    parser.add_argument("--ref-image", type=Path, default=None, help="Optional portrait image (<Picture 1>) for MMH3")
    parser.add_argument("--drama3-url", default=f"http://{os.getenv('DRAMA3_HOST', '127.0.0.1')}:{os.getenv('DRAMA3_PORT', '8013')}/v1/audio/speech")
    parser.add_argument("--mmh3-bin", type=Path, default=Path(os.getenv("MMH3_BIN", "./mmh3/target/release/mmh3")))
    parser.add_argument("--mmh3-models", type=Path, default=Path(os.getenv("MMH3_MODELS_DIR", "./mmh3/models")))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path(os.getenv("PANEL_OUTPUT_DIR", "./outputs")) / "starter_out.mp4")
    parser.add_argument("--remux-clean-audio", action="store_true", help="Remux original Drama-3 WAV onto final MP4")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved commands without executing MMH3")
    args = parser.parse_args()

    presets = load_presets(args.presets_file)
    if args.preset not in presets:
        raise SystemExit(f"Unknown preset '{args.preset}'. Available: {', '.join(sorted(presets))}")
    preset = presets[args.preset]
    mmh3_cfg = preset.get("mmh3", {})

    tts_text = build_tts_text(args.dialogue, preset)
    visual_prompt = build_video_prompt(
        args.scene_prompt,
        preset,
        has_reference_image=args.ref_image is not None,
    )

    work_dir = args.out.parent / f".{args.out.stem}_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    speech_wav = work_dir / "drama3_speech.wav"
    raw_mp4 = work_dir / "mmh3_raw.mp4"

    if args.dry_run:
        duration_s = 5.0
        print(f"[DRY-RUN] Drama-3 URL : {args.drama3_url}")
        print(f"[DRY-RUN] TTS Text    : {tts_text}")
    else:
        print(f"[1/3] Synthesizing speech via Drama-3 ({args.drama3_url})...")
        duration_s = synthesize_drama3(
            args.drama3_url,
            tts_text,
            speech_wav,
            ref_audio=args.ref_voice,
            ref_text=args.ref_voice_text,
        )
        print(f"      Saved {speech_wav} ({duration_s:.2f}s)")

    fps = int(mmh3_cfg.get("fps", 24))
    # Align frame count cleanly to 4k+1 temporal latent boundary (min 25 frames)
    raw_frames = max(25, int(math.ceil(duration_s * fps)))
    frames = ((raw_frames - 1 + 3) // 4) * 4 + 1

    cmd = [
        str(args.mmh3_bin),
        "generate",
        "--models",
        str(args.mmh3_models),
        "--out",
        str(raw_mp4),
        "--prompt",
        visual_prompt,
        "--width",
        str(mmh3_cfg.get("width", 768)),
        "--height",
        str(mmh3_cfg.get("height", 1280)),
        "--frames",
        str(frames),
        "--steps",
        str(mmh3_cfg.get("steps", 24)),
        "--seed",
        str(args.seed),
        "--reference-audio",
        str(speech_wav),
    ]
    if mmh3_cfg.get("lock_audio", True):
        cmd.append("--lock-audio")
    if args.ref_image is not None:
        cmd += ["--reference", str(args.ref_image)]
    for extra in mmh3_cfg.get("extra_args", []):
        cmd.append(str(extra))

    if args.dry_run:
        print(f"[DRY-RUN] Video Prompt: {visual_prompt}")
        print(f"[DRY-RUN] MMH3 Cmd    : {' '.join(cmd)}")
        return 0

    print(f"[2/3] Generating audio-locked video via MMH3 ({frames} frames @ {fps}fps)...")
    subprocess.run(cmd, check=True)

    print(f"[3/3] Post-processing -> {args.out}...")
    final_path = run_post_process(
        raw_mp4,
        speech_wav,
        args.out,
        preset,
        remux_audio=args.remux_clean_audio,
        strip_metadata=True,
    )
    print(f"[DONE] Final output: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
