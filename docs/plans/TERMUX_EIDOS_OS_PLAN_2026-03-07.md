# Termux Eidos OS Plan

Date: 2026-03-07

Purpose:
- redesign the Termux startup/bootstrap path
- make `eidosian_venv` the default active Python environment in a controlled, portable way
- unify services, models, dashboard control, X11, and forge automation into one coherent user-space system
- preserve rollback safety and Linux portability

## Current State

Verified locally:
- `eidosian_venv/bin` is missing the standard `activate` scripts entirely.
- `~/.bashrc` is overloaded:
  - X11 startup
  - notification logic
  - aliases
  - custom PATH manipulation
  - Eidos service bootstrap
  - Eidos environment source
  - unrelated legacy home-script automation
- `PATH` currently resolves `~/scripts` before forge commands, which already caused `qwenchat` shadowing.
- `eidos_env.sh` is modular and cleaner than `~/.bashrc`, but only auto-activates in forge directories unless `EIDOS_AUTO_ACTIVATE_FORGE=1`.
- `.termux/termux.properties` is mostly default and does not yet enable external command control.
- `~/.termux/boot/` does not exist yet.
- X11 is already being launched from `~/.bashrc` through `~/scripts/start_x11`.
- Eidos services are started from `~/.bashrc`, not from a Termux-native boot/service stack.

## Design Goals

1. Keep shell startup fast, deterministic, and reversible.
2. Move long-running services out of `~/.bashrc` where possible.
3. Make `eidosian_venv` the default Python/pip toolchain for interactive Termux use without breaking non-forge shells.
4. Preserve one canonical service registry:
   - ports
   - service identities
   - model roles
   - logs
   - health checks
5. Make the web dashboard a real control plane for:
   - services
   - pipeline state
   - graph/lexicon/code exploration
   - shell command execution via guarded APIs
6. Keep Linux portability first-class:
   - same scripts
   - same env contracts
   - same service registry
   - different platform adapters
7. Avoid proot and distro installs.
8. Exploit Android-native affordances where they are actually useful:
   - Termux:API
   - RUN_COMMAND
   - Boot
   - Widget
   - Tasker
   - Float
   - X11

## Guiding Constraints

From official Termux docs:
- All Termux plugins must come from the same signing source as the main app.
- `RUN_COMMAND` requires `com.termux.permission.RUN_COMMAND` and `allow-external-apps`.
- `termux-services` expects shell restart so the service-daemon starts.
- Termux boot automation uses `~/.termux/boot/` and commonly starts with `termux-wake-lock`.
- Termux execution environment has Android-specific `PATH`, `LD_PRELOAD`, and exec restrictions; system binaries should be called with sanitized environment when needed.

## Target Architecture

### Layer 0: Minimal Shell Entry

`~/.bashrc` should become a thin dispatcher:
- shell cosmetics
- load shared shell utility library
- source one canonical Eidos bootstrap
- no direct long-running process startup except the smallest service handoff

Target:
- 1 small bootstrap block
- 1 legacy-compat section
- everything else moved to modular files under `eidosian_forge/shell/`

### Layer 1: Cross-Platform Environment Core

Create a platform-aware env core:
- `shell/env.d/`
- `shell/platforms/termux.sh`
- `shell/platforms/linux.sh`

Responsibilities:
- detect Termux vs Linux
- define safe PATH/PYTHONPATH/LD_LIBRARY_PATH composition
- define sanitized system-command execution
- define platform-specific helpers

This layer must restore classic venv ergonomics:
- regenerate or provide portable activation scripts
- provide `activate`, `deactivate`, `forge-env`, `forge-reset`, `forge-python`, `forge-pip`
- optionally symlink or wrap them under `~/bin` for portability

### Layer 2: Service Substrate

Unify around a single service model:
- keep `config/ports.json` canonical
- keep coordinator canonical for model/resource scheduling
- move always-on daemons to Termux-native service supervision

Preferred structure:
- `termux-services` / runit for steady-state daemons
- `~/.termux/boot/` for device boot entrypoint
- shell startup only ensures user-session services are connected, not launched ad hoc every time

Service classes:
- core HTTP:
  - MCP
  - Atlas
  - dashboard APIs
- model:
  - qwen Ollama
  - embedding Ollama
- pipeline:
  - scheduler
  - doc processing worker
  - graph/vector maintenance
- UI:
  - X11 launcher/session manager

### Layer 3: Eidos Control Plane

Atlas evolves into a Termux control plane:
- service start/stop/restart
- health and logs
- scheduler queue and ETA
- coordinator state
- X11 session status
- device metrics
- direct command execution via guarded backend

Control mechanisms:
- local Python backend first
- optional Android-side launch/control via `RUN_COMMAND`
- optional widget/tasker triggers for mobile UX

### Layer 4: Eidos OS User-Space

The user-visible system should feel like a coherent OS shell:
- command palette
- dashboard
- graph explorer
- living lexicon
- code library browser
- docs browser
- shell task control
- Android integration

## Detailed Phased Plan

## Phase 0: Baseline, Rollback, and Safety

Deliverables:
- snapshot current:
  - `~/.bashrc`
  - `~/.termux/termux.properties`
  - `~/scripts/`
  - Eidos shell modules
  - startup logs
- create restore scripts:
  - `scripts/termux_backup_startup.sh`
  - `scripts/termux_restore_startup.sh`
- create startup audit:
  - PATH order
  - duplicate aliases
  - duplicate service launches
  - shell startup duration

Acceptance:
- one-command rollback to previous shell startup
- startup diff report generated before edits

## Phase 1: Rebuild Shell Bootstrap

Deliverables:
- replace long `~/.bashrc` Eidos section with:
  - `source ~/eidosian_forge/eidos_env.sh`
  - `eidos-shell-start`
- create:
  - `shell/profile.d/termux_interactive.sh`
  - `shell/profile.d/linux_interactive.sh`
  - `shell/profile.d/prompt.sh`
  - `shell/profile.d/path.sh`
  - `shell/profile.d/aliases.sh`
- isolate legacy home script aliases into:
  - `shell/legacy/home_scripts.sh`

Important change:
- `PATH` must put forge wrappers before `~/scripts` for forge-owned commands.
- legacy home scripts should either be namespaced or explicitly wrapped.

Acceptance:
- no command shadowing for Eidos-owned tools
- shell startup remains correct in:
  - `~`
  - forge repo
  - non-forge project dirs

## Phase 2: Restore Proper Python Activation

Problem:
- standard venv activation scripts are missing.

Deliverables:
- restore generated `activate`, `activate.fish`, `activate.csh`
  or provide maintained equivalents
- add:
  - `scripts/repair_venv_activation.sh`
  - `scripts/forge_activate.sh`
- set interactive default so:
  - `python`
  - `pip`
  - `pytest`
  - `uvicorn`
  - forge CLIs
  come from `eidosian_venv`

Preferred model:
- globally default to `eidosian_venv` in Termux interactive shells
- provide explicit `system-python` and `system-pip` escape hatches

Acceptance:
- `which python` points to forge venv in interactive Termux shell
- Linux shell gets the same behavior through the same bootstrap layer
- deactivation/reset remains possible

## Phase 3: Move to Native Termux Service Management

Deliverables:
- install/use `termux-services`
- create runit services for:
  - Ollama qwen
  - Ollama embedding
  - MCP
  - Atlas
  - scheduler
  - doc worker
- keep custom `eidos_termux_services.sh` as compatibility/orchestration wrapper, but backed by runit

Desired state:
- shell startup does not spawn services directly
- services are supervised independently
- logs live in predictable places
- service status uses `sv` and dashboard APIs

Acceptance:
- `sv status` reflects all core daemons
- service restart no longer depends on shell lifecycle

## Phase 4: Boot and Resume Semantics

Deliverables:
- create `~/.termux/boot/` scripts
- first boot script:
  - `termux-wake-lock`
  - source `start-services.sh`
  - warm core services in correct order
- boot-safe resume logic for:
  - scheduler
  - code ingestion
  - doc generation
  - GraphRAG indexing
  - lexicon queue processing

Acceptance:
- reboot resumes services and pipeline state safely
- no duplicate launches after manual shell start

## Phase 5: X11 Session Manager

Current state:
- X11 is launched ad hoc from `~/.bashrc`.

Deliverables:
- replace `~/scripts/start_x11` ad hoc launch with:
  - runit-supervised X11 session helper
  - explicit profile and display config
  - logs and health checks
- add profile presets:
  - minimal desktop
  - window-manager-only
  - web dashboard kiosk
  - mixed shell + Atlas control
- add compatibility knobs:
  - `-legacy-drawing`
  - `-force-bgra`
  - `TERMUX_X11_XSTARTUP`

Acceptance:
- X11 can be controlled from dashboard and shell
- X11 no longer creates duplicate launches

## Phase 6: Dashboard as Termux Control Plane

Deliverables:
- extend Atlas with:
  - service management pane
  - shell execution pane
  - log viewer
  - job queue / ETA view
  - X11 session control
  - Android API action pane
- define safe execution layers:
  - read-only status APIs
  - guarded command APIs
  - high-risk commands only through explicit approval or signed action policy

Potential Termux-specific control modes:
- direct local subprocess
- `RUN_COMMAND` integration for Android-launchable actions
- `Termux:Widget` shortcuts for dashboard actions
- `Tasker` automation for phone-native triggers

Acceptance:
- dashboard can manage core user-space without requiring manual shell juggling

## Phase 7: Termux UX and Styling

Deliverables:
- redesign `.termux/termux.properties`
  - transcript rows
  - cursor style/blink
  - fullscreen
  - keyboard behavior
  - extra keys
  - margins
  - bell/back-key behavior
- define opinionated Eidos profile sets:
  - touch-first
  - hardware-keyboard
  - dashboard-focused
  - coding-focused
- optionally standardize on `Termux:Styling`
  - font
  - theme
  - Nerd Font support where appropriate

Acceptance:
- one command to switch profile
- consistent look across shell, X11, and dashboard branding

## Phase 8: Android Integration Layer

Deliverables:
- package/use `termux-api`
- create Eidos wrappers for:
  - notifications
  - clipboard
  - storage picker/share
  - media/session state
  - sensors/battery/network if useful
- optional integrations:
  - `Termux:Float` for quick shell overlay
  - `Termux:Widget` for launcher shortcuts
  - `Termux:Tasker` for event-driven automations

Acceptance:
- Android-native affordances are routed through one Eidos adapter layer
- Linux path gets graceful no-op or equivalent replacements

## Phase 9: Coordinator and Model Governance

Deliverables:
- every model-using forge must register with the coordinator
- model leases:
  - shared embed
  - exclusive qwen
  - interactive override
  - scheduler priority
- `qwenchat` behavior:
  - reads current ETA
  - requests exclusive lease
  - blocks background qwen work
  - releases on exit

Additional work:
- scheduler should actively pause/defer qwen-heavy phases when interactive lease is active
- embed tasks may continue if they are explicitly allowed by policy

Acceptance:
- no accidental concurrent qwen usage
- user-facing ETA before interactive takeover

## Phase 10: Files, Scripts, and Portability

Deliverables:
- audit `~/scripts/` into categories:
  - keep
  - namespace
  - deprecate
  - absorb into forge
- move reusable tools into cross-platform wrappers
- define a portable `~/bin/eidos-*` surface
- create Linux/Termux compatibility helpers for:
  - notifications
  - clipboard
  - app launching
  - wakelock
  - Android intents

Acceptance:
- same command names work on Linux and Termux where possible
- platform-specific differences are hidden behind adapters

## Phase 11: Performance and Reliability

Deliverables:
- shell startup benchmark
- service cold-start benchmark
- qwen warmup and request latency benchmark
- X11 launch benchmark
- dashboard load benchmark
- memory/CPU/battery profile

Specific fixes to pursue:
- stop running too much directly in shell init
- keep system-command execution environment sanitized
- reduce duplicate background processes
- move heavy warmups to supervised services
- prewarm qwen and embedding models intelligently, not blindly

Acceptance:
- measurable startup regression budget
- deterministic logs for all daemons

## Phase 12: Eidos OS Surface

Deliverables:
- coherent naming and UX:
  - shell
  - dashboard
  - services
  - graph explorer
  - lexicon
  - docs
  - code library
- session persistence
- live state panels
- mobile-friendly control flows

Acceptance:
- user can manage the whole Eidos stack from either shell or dashboard

## Recommended Execution Order

1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. Phase 4
6. Phase 9
7. Phase 5
8. Phase 6
9. Phase 7
10. Phase 8
11. Phase 10
12. Phase 11
13. Phase 12

Reason:
- bootstrap correctness first
- then supervised services
- then model governance
- then UI/control plane expansion

## Immediate Next Implementation Slice

Highest-value first slice:
- restore venv activation scripts
- make forge Python the default interactive Python
- shrink `~/.bashrc` to thin bootstrap
- create `~/.termux/boot/`
- migrate current always-on services to `termux-services`

This solves the biggest current pain points:
- missing activation
- brittle startup
- PATH shadowing
- duplicated service management

## Rollback Strategy

Every phase should include:
- pre-change backup
- generated diff report
- one restore command
- one validation command

Rollback command pattern:
- `scripts/termux_restore_startup.sh <snapshot-id>`

## Reference Set

Saved local copies:
- `docs/external_references/2026-03-07-termux-upgrade/`

Primary upstream sources:
- Termux app
- RUN_COMMAND intent docs
- Termux execution environment wiki
- Termux file system layout wiki
- termux-services
- termux-x11
- termux-boot
- termux-styling
- termux-float
- termux-tasker
- termux-widget
- termux-api-package
