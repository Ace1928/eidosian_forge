# Data Governance

## Policy
- `data/` is allowed for tracked, reviewable project knowledge/state.
- Runtime-only generated artifacts are explicitly ignored in `.gitignore`.
- Secrets must never be committed in `data/` or any tracked path.

## Controls
1. **Pre-commit**
- `gitleaks` hook scans staged content using `.gitleaks.toml`.
- `detect-private-key` hook remains enabled.

2. **CI**
- `.github/workflows/secret-scan.yml` runs `gitleaks` on PRs and pushes to `main`.

3. **Scoped Ignore Rules**
- Runtime outputs are ignored by path patterns (e.g. `data/code_forge/roundtrip/`, sqlite artifacts, caches, temp/event stores).
- Curated data (knowledge/memory JSON, governed config) remains trackable.

## Contributor Rules
- Do not commit `.env` files, tokens, API keys, credentials, or private keys.
- If test fixtures require secret-like strings, use obvious placeholders (`example_`, `dummy_`, `test_`, `fake_`) so allowlist rules can remain narrow.
- Prefer deterministic JSON artifacts with explicit schemas for tracked data.

## Incident Response
If a secret is detected:
1. Revoke/rotate the credential immediately.
2. Remove the secret from working tree and commit history as needed.
3. Re-run secret scan locally and in CI.
4. Record remediation in security audit docs/issues.
