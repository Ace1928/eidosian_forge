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

# Sanitize a local text file
python moltbook_forge/moltbook_sanitize.py --input /path/to/post.txt

# Ingest from a URL (requires --allow-network and allowlisted domains)
python moltbook_forge/moltbook_ingest.py --url https://example.com/feed.json --allow-network
```

## Safety Notes

- All inputs are treated as untrusted data.
- Any suspicious content is flagged for manual review.
- Sanitized output is JSON for deterministic downstream handling.
