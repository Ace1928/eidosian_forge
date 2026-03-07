# Eidosian Shell Modules

`shell/bootstrap.sh` is the canonical interactive shell loader. Source it from
`~/.bashrc` on Termux or Linux.

`shell/termux_bootstrap.sh` is the Termux-specific layer loaded by
`shell/bootstrap.sh` when Termux is detected.

`eidos_env.sh` remains the forge environment entry point used by the shell
bootstrap to activate the forge venv and path contract.

Common shell modules live in `shell/common.d` and are loaded in lexical order:

- `00_common_helpers.sh`: logging, notifications, platform checks, path helpers, and utility functions
- `10_common_runtime.sh`: history, shell options, portable path setup, and `.env` sourcing
- `20_common_aliases.sh`: portable aliases and script launch aliases
- `30_common_prompt.sh`: prompt initialization and command-not-found behavior

Forge environment modules live in `shell/env.d` and are loaded by `eidos_env.sh`:

- `00_helpers.sh`: path normalization and de-duplication helpers
- `10_baseline.sh`: capture and sanitize the user's baseline shell environment
- `20_environment.sh`: forge activation, reset, and safe execution helpers
- `30_commands.sh`: command wrappers and aliases
- `40_runtime.sh`: auto-activation policy and interactive notifications

Termux session modules live in `shell/profile.d` and are loaded by
`shell/termux_bootstrap.sh`:

- `00_termux_helpers.sh`: Termux/platform helper functions
- `10_termux_runtime.sh`: PulseAudio, X11, notifications, storage aliases, and Termux build flags
- `20_eidos_bootstrap.sh`: forge env bootstrap and service manager handoff
- `30_files_dashboard.sh`: files dashboard aliases and autostart
- `40_npm_completion.sh`: npm completion setup

Install helpers:

- `scripts/install_shell_bootstrap.sh`: installs the thin `~/.bashrc` wrapper from the repo template
- `scripts/install_shell_prereqs.sh`: installs recommended shell packages on Termux/Linux
- `scripts/install_termux_boot.sh`: installs `~/.termux/boot/00-eidos-boot`
- `scripts/eidos_termux_boot.sh`: boot-safe orchestration entrypoint for Termux:Boot
- `scripts/repair_venv_activation.sh`: restores standard venv activate scripts if needed
- `scripts/termux_backup_startup.sh`: snapshot current startup files before changes
- `scripts/termux_restore_startup.sh`: restore a startup snapshot
- `scripts/termux_audit_startup.py`: emit a machine-readable startup audit

Design constraints:

- keep `~/.bashrc` thin and repo-managed
- never install packages implicitly during shell startup
- keep the forge venv activation deterministic and reversible
- keep boot orchestration separate from interactive shell startup
- preserve portability so the same bootstrap can run on Termux or Linux where possible
- use `scripts/eidos_safe_run.sh` for long or risky jobs instead of ambient shell state
