"""Default pass-through prompt & dialogue hook.

Replace or wrap `build_tts_text` and `build_video_prompt` in your local
environment to plug in custom dialogue normalization, emotion taggers, or
structured scene-prompt compilers.
"""
from __future__ import annotations

from typing import Any


def build_tts_text(raw_text: str, preset: dict[str, Any] | None = None) -> str:
    """Normalize dialogue text sent to Drama-3."""
    _ = preset
    return " ".join(raw_text.strip().split())


def build_video_prompt(
    scene_prompt: str,
    preset: dict[str, Any] | None = None,
    *,
    has_reference_image: bool = False,
) -> str:
    """Assemble the final visual prompt passed to mmh3."""
    parts: list[str] = []
    if preset:
        prefix = str(preset.get("prompt_prefix", "")).strip()
        if prefix:
            parts.append(prefix)

    cleaned = scene_prompt.strip()
    if has_reference_image and "<Picture 1>" not in cleaned:
        cleaned = f"<Picture 1> {cleaned}" if cleaned else "<Picture 1> speaks naturally to camera."
    if cleaned:
        parts.append(cleaned)

    if preset:
        suffix = str(preset.get("prompt_suffix", "")).strip()
        if suffix:
            parts.append(suffix)

    return " ".join(parts)
