"""Default post-processing hook (clean ffmpeg mux + metadata strip).

Extend `run_post_process` locally for custom color grading, loudness mastering,
frame interpolation, or automated candidate QA scoring.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any


def run_post_process(
    video_path: Path,
    audio_path: Path | None,
    output_path: Path,
    preset: dict[str, Any] | None = None,
    *,
    remux_audio: bool = False,
    strip_metadata: bool = True,
) -> Path:
    """Finalize output video with optional clean audio remux & metadata stripping."""
    _ = preset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        if video_path.resolve() != output_path.resolve():
            shutil.copy2(video_path, output_path)
        return output_path

    cmd = [ffmpeg, "-y", "-i", str(video_path)]
    if remux_audio and audio_path is not None and audio_path.exists():
        cmd += [
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
        ]
    else:
        cmd += ["-c", "copy"]

    if strip_metadata:
        cmd += ["-map_metadata", "-1", "-fflags", "+bitexact"]

    cmd.append(str(output_path))
    subprocess.run(cmd, check=True)
    return output_path
