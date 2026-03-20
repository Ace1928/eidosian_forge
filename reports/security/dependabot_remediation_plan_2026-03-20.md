# Dependabot Remediation Batches

- generated_at: `2026-03-20T08:00:59Z`
- repo: `Ace1928/eidosian_forge`
- batch_count: `3`
- total_batched_alerts: `11`

## Batches

| Key | Severity | Alerts | Ecosystem | Manifest |
|---|---:|---:|---|---|
| `pip::requirements-eidos-venv-reqs-txt` | `critical` | `8` | `pip` | `requirements/eidos_venv_reqs.txt` |
| `pip::requirements-eidosian-venv-reqs-txt` | `high` | `2` | `pip` | `requirements/eidosian_venv_reqs.txt` |
| `npm::game-forge-src-autoseed-package-lock-json` | `high` | `1` | `npm` | `game_forge/src/autoseed/package-lock.json` |

### `pip::requirements-eidos-venv-reqs-txt`

- max_severity: `critical`
- total_alerts: `8`
- package_count: `5`
- suggested_branch: `security/remediation/pip-requirements-eidos-venv-reqs-txt-critical`
- suggested_pr_title: `security: remediate critical deps in requirements/eidos_venv_reqs.txt`

Top packages:
- `authlib` x `3`
- `ujson` x `2`
- `pyasn1` x `1`
- `pyopenssl` x `1`
- `PyJWT` x `1`

### `pip::requirements-eidosian-venv-reqs-txt`

- max_severity: `high`
- total_alerts: `2`
- package_count: `2`
- suggested_branch: `security/remediation/pip-requirements-eidosian-venv-reqs-txt-high`
- suggested_pr_title: `security: remediate high deps in requirements/eidosian_venv_reqs.txt`

Top packages:
- `nltk` x `1`
- `PyJWT` x `1`

### `npm::game-forge-src-autoseed-package-lock-json`

- max_severity: `high`
- total_alerts: `1`
- package_count: `1`
- suggested_branch: `security/remediation/npm-game-forge-src-autoseed-package-lock-json-high`
- suggested_pr_title: `security: remediate high deps in game_forge/src/autoseed/package-lock.json`

Top packages:
- `flatted` x `1`

