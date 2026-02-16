# Part 08: Termux and Linux Hardening

## Goal

Ensure operational parity and robust fallback behavior across environments.

## Hardening Tasks

1. Validate import path setup and module discovery.
2. Validate file permissions and append-only event writes.
3. Verify subprocess/scheduler behavior in Android constraints.
4. Provide fallback for missing optional dependencies.

## Acceptance Criteria

- Boot and diagnostics run cleanly under Termux venv.
- Linux parity smoke tests pass.
