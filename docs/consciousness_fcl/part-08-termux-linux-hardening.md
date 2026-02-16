# Part 08: Termux and Linux Hardening

## Goal

Ensure operational parity and robust fallback behavior across environments.

## Hardening Tasks

1. ~~Validate import path setup and module discovery.~~
2. ~~Validate file permissions and append-only event writes.~~
3. ~~Verify subprocess/scheduler behavior in Android constraints.~~
4. ~~Provide fallback for missing optional dependencies.~~
5. ~~Harden Tika URL extraction for missing `from_url` API and Termux SSL-chain failures.~~
6. ~~Add Linux parity smoke automation script and CI workflow.~~
7. [ ] Validate Linux CI runner parity against the same audit matrix.
8. ~~Ensure full MCP integration tests backup/restore mutable KB and memory files so repeated test runs are idempotent.~~

## Acceptance Criteria

- Boot and diagnostics run cleanly under Termux venv.
- Linux parity smoke tests pass.
- Latest Termux audit: `./scripts/termux_audit.sh` PASS on `2026-02-16`.
