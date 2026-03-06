# Eidosian Shell Modules

`eidos_env.sh` is the bootstrap entry point sourced from `~/.bashrc`.

`shell/termux_bootstrap.sh` is the higher-level Termux session loader sourced
from `~/.bashrc`. It handles interactive Termux runtime concerns that should
not live inline in the shell rc file:

- PulseAudio startup
- X11 startup
- Eidos service bootstrap
- files dashboard aliases/autostart
- npm completion
- Termux-specific build flags

Modules live in `shell/env.d` and are loaded in lexical order:

- `00_helpers.sh`: path normalization and de-duplication helpers
- `10_baseline.sh`: capture and sanitize the user's baseline shell environment
- `20_environment.sh`: forge activation, reset, and safe execution helpers
- `30_commands.sh`: command wrappers and aliases
- `40_runtime.sh`: auto-activation policy and interactive notifications

Termux session modules live in `shell/profile.d` and are loaded by
`shell/termux_bootstrap.sh`:

- `00_termux_helpers.sh`: Termux/platform helper functions
- `10_termux_runtime.sh`: PulseAudio, X11, notifications, and shared aliases
- `20_eidos_bootstrap.sh`: forge env bootstrap and service manager handoff
- `30_files_dashboard.sh`: files dashboard aliases and autostart
- `40_npm_completion.sh`: npm completion setup

Design constraints:

- Outside `eidosian_forge`, keep the user's shell baseline clean.
- Inside `eidosian_forge`, activate the forge environment deterministically.
- For long or risky jobs, use `scripts/eidos_safe_run.sh` so execution happens in a sanitized environment instead of the ambient shell state.
