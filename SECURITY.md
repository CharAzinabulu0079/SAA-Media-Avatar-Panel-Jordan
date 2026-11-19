# Security & Responsible Use Policy

## 1. Zero Credentials & Local Isolation

This repository contains **engine patches, configuration templates, and modular starter scripts only**. It contains **zero API keys, tokens, hostnames, private IP addresses, voice samples, or model weights**.

## 2. Production Serving Hygiene

1. **Loopback-Only Binding**: Always bind local inference engines (`drama-tensorfold`, `drama3_bridge_starter.py`, and `mmh3 serve`) to `127.0.0.1` by default. Place an authenticated reverse proxy in front of any service that needs remote or LAN access.
2. **Unified-Memory Coexistence Guard (NVIDIA DGX Spark GB10)**: On unified-memory systems (`121.7 GiB` shared CPU/GPU pool), always leave OS headroom and avoid starting heavy diffusion runs while an unguarded full-memory LLM or training job is saturating `MemAvailable`.
3. **Output Sanitization**: When sharing generated media publicly, strip unintended container metadata if your workflow injects local filesystem paths.

## 3. Responsible Voice & Media Generation

When running Fish Audio Drama 3 checkpoints (including `ThinkingVibeCoder/Drama-3-Tier2-5Lang-INT3`) or MiniMax H3 checkpoints:

- Comply with the **Fish Audio Community License v1.0 (Section IV.b)** and **MiniMax H3 Community License** terms.
- Never clone or imitate the voice or likeness of any identifiable real person without explicit prior consent or lawful authorization.
- Never remove, disable, or circumvent model provenance signals or watermarks.
