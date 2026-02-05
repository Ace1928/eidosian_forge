# Moltbook Forge

Safe ingestion tooling for Moltbook (social network for AI).

This forge is designed to *quarantine and normalize* untrusted content before it
can influence memory, tasks, or code. No network access is performed unless
explicitly allowed.

## Goals

- Strong normalization and standardization of text inputs.
- Heuristic prompt-injection detection and flagging.
- Explicit allowlist for network sources.
- Zero code execution from remote content.

## Usage

```bash
source /home/lloyd/eidosian_forge/eidosian_venv/bin/activate

# List CLI commands
python -m moltbook_forge.cli --list

# Sanitize a local text file
python moltbook_forge/moltbook_sanitize.py --input /path/to/post.txt

# Screen a sanitized payload
python moltbook_forge/moltbook_screen.py --input sanitized.json --threshold 0.4

# Validate a sanitized payload
python moltbook_forge/moltbook_validate.py --input sanitized.json

# Quarantine if needed
python moltbook_forge/moltbook_quarantine.py --input sanitized.json --threshold 0.4

# Bootstrap a Moltbook skill snapshot (no execution)
python moltbook_forge/moltbook_bootstrap.py --input skill.md --output-dir moltbook_forge/skill_sources

# Review a skill file through the safety pipeline
python moltbook_forge/moltbook_skill_review.py --input skill.md --output skill_report.json
# Exit codes: 0=allow, 1=invalid, 2=quarantine

# Ingest from a URL (requires --allow-network and allowlisted domains)
python moltbook_forge/moltbook_ingest.py --url https://moltbook.com/feed.json --allow-network
```

## Safety Notes

- All inputs are treated as untrusted data.
- Any suspicious content is flagged for manual review.
- Sanitized output is JSON for deterministic downstream handling.
