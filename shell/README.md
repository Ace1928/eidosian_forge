# Eidosian Shell Modules

`eidos_env.sh` is the bootstrap entry point sourced from `~/.bashrc`.

Modules live in `shell/env.d` and are loaded in lexical order:

- `00_helpers.sh`: path normalization and de-duplication helpers
- `10_baseline.sh`: capture and sanitize the user's baseline shell environment
- `20_environment.sh`: forge activation, reset, and safe execution helpers
- `30_commands.sh`: command wrappers and aliases
- `40_runtime.sh`: auto-activation policy and interactive notifications

Design constraints:

- Outside `eidosian_forge`, keep the user's shell baseline clean.
- Inside `eidosian_forge`, activate the forge environment deterministically.
- For long or risky jobs, use `scripts/eidos_safe_run.sh` so execution happens in a sanitized environment instead of the ambient shell state.
