# Scripts Library

A modular, automation-friendly collection of local scripts.

## Conventions
- All scripts support `-h/--help` and stable exit codes.
- Scripts are non-interactive by default when flags are provided.
- Config overrides use environment variables or `~/.config/eidosian/<script>.env`.

## Index
- `add_script`: save clipboard/stdin as an executable script
- `add_to_path`: add a directory to PATH
- `clipboard_diagnostics.sh`: detect clipboard backends
- `completion_install`: install shell completions from executables
- `download_local_models.py`: curated Hugging Face model downloader using `config/model_catalog.json`
- `forge_builder`: create project structures
- `generate_directory_atlas.py`: generate linked directory atlas + full recursive index (deterministic tracked scope by default; optional filesystem scope)
- `glyph_stream`: render images/video to glyph art
- `interactive_delete`: safe interactive deletion helper
- `overwrite`: overwrite a file from clipboard/stdin
- `pdf_check_text`: check if PDF contains text
- `pdf_check_encryption`: check if PDF is encrypted
- `pdf_to_text`: convert PDF to text
- `pdf_tools_menu`: GUI menu for PDF tools
- `smart_publish`: publish Python packages with version bumping
- `treegen`: generate a file tree report

## Docs
- `docs/prompts/eidosian_refinement_prompt.txt`: refinement prompt text

## Common flags
- `-h, --help`: usage
- `-q, --quiet`: reduce output
- `-v, --verbose`: increase output
- `--json`: machine-readable output (where supported)

## Config
- Default config directory: `~/.config/eidosian/`
- Script-specific env file: `~/.config/eidosian/<script>.env`

## Testing
- Python tests: `pytest -q`
- Bash checks: run scripts with `--help` and `--dry-run`
