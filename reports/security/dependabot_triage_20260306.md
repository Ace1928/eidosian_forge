# Dependabot Triage 2026-03-06

## Scope

- Source of truth: GitHub Dependabot alerts for `Ace1928/eidosian_forge`
- Verification tools:
  - `gh api /repos/Ace1928/eidosian_forge/dependabot/alerts?state=open&per_page=100`
  - `eidosian_venv/bin/python -m pip_audit --local`
  - `npm audit --json` in `game_forge/src/autoseed`

## Findings Before Remediation

- `requirements/eidos_venv_reqs.txt`
  - `Authlib==1.6.6` -> fixed to `1.6.7`
  - `Markdown==3.7` -> fixed to `3.8.1`
  - `pypdf==6.7.1` -> fixed to `6.7.5`
  - `yt-dlp==2025.3.26` -> fixed to `2026.02.21`
- `doc_forge/requirements.txt`
  - `Markdown==3.7` -> fixed to `3.8.1`
- `doc_forge/docs/requirements.txt`
  - `Markdown==3.7` -> fixed to `3.8.1`
- `game_forge/src/autoseed/package-lock.json`
  - transitive `minimatch` high-severity alerts
  - transitive `rollup` high-severity alert
  - `npm audit fix --package-lock-only` remediated the lockfile

## Validation After Remediation

- `pip-audit --local`: no known vulnerabilities in the active `eidosian_venv`
- `npm audit --json`: `0` vulnerabilities in `game_forge/src/autoseed`

## Notes

- GitHub alert counts may remain open until Dependabot reprocesses the updated manifests and lockfile on the default branch.
- `pip-audit -r requirements/eidosian_venv_reqs.txt` is not reliable on this Termux path because isolated resolution hits Android API detection failures for Rust-backed wheels. `--local` is the correct verification mode for this environment.
