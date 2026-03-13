# `glyph_forge/tests`

<!-- EIDOS:DOCSYS:START -->
- Contract: `eidos.directory_doc.v1`
- Generated: `2026-03-13T00:19:22Z`
- Path: `glyph_forge/tests`

## What It Is

Managed directory documentation for `glyph_forge/tests`, generated from tracked files and directory structure.

## Why It Exists

This directory exists to hold the `tests` surface for `glyph_forge` and keep that responsibility separate from adjacent forge concerns.

## How It Works

- Tracked files in scope: `24`
- Child directories: `1`
- Tests detected: `True`
- Python modules: `conftest, helpers_fakes, test_api, test_audio_tools, test_auto_function_coverage, test_banner, test_batch_processing, test_cli_targets, test_core_streaming, test_eidos_profile`
- Detected API routes: `PATCH PIL.Image.open; PATCH builtins.open; PATCH glyph_forge.services.image_to_glyph.ThreadPoolExecutor; PATCH glyph_forge.services.image_to_glyph.shutil.get_terminal_size; PATCH os.makedirs; PATCH shutil.get_terminal_size; PATCH subprocess.Popen`

## Contents

- [`benchmarks`](./benchmarks/README.md)

## Prominent Files

- [`conftest.py`](./conftest.py)
- [`helpers_fakes.py`](./helpers_fakes.py)
- [`test_api.py`](./test_api.py)
- [`test_audio_tools.py`](./test_audio_tools.py)
- [`test_auto_function_coverage.py`](./test_auto_function_coverage.py)
- [`test_banner.py`](./test_banner.py)
- [`test_batch_processing.py`](./test_batch_processing.py)
- [`test_cli_targets.py`](./test_cli_targets.py)

## API Surface

- `PATCH PIL.Image.open`
- `PATCH builtins.open`
- `PATCH glyph_forge.services.image_to_glyph.ThreadPoolExecutor`
- `PATCH glyph_forge.services.image_to_glyph.shutil.get_terminal_size`
- `PATCH os.makedirs`
- `PATCH shutil.get_terminal_size`
- `PATCH subprocess.Popen`

## Strengths

- A directly associated test surface is present in or below this directory.
- The directory contains detected HTTP/API route definitions that can be referenced programmatically.
- The directory exposes importable Python modules rather than only opaque assets.
- Responsibility is split into child directories instead of one flat file heap.

## Weaknesses / Risks

- No structural documentation risks were detected automatically; functional review is still required for behavior-level claims.

## Next Steps

- Keep this README synchronized with code and test changes through the managed documentation toolchain.

## References

- Parent README: [`glyph_forge/README.md`](../README.md)

## Accuracy Contract

- This README is generated from tracked repository structure and conservative code scanning.
- Behavior claims should be backed by tests, routes, or directly linked source files.
<!-- EIDOS:DOCSYS:END -->
