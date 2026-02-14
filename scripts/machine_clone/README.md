# Machine Clone Tooling

This directory contains operational scripts for:

- exporting a host profile
- importing curated legacy projects
- building a dual-live + encrypted vault USB
- restoring a new host from profile artifacts
- verifying host parity after restore
- enrolling a node into the Eidos sync mesh
- running post-boot onboarding checks on the restored host
- validating Moltbook credential continuity and API access after restore

## Scripts

- `export_host_profile.sh`
- `import_legacy_projects.sh`
- `build_dual_live_usb.sh`
- `repair_dual_live_usb.sh`
- `verify_dual_live_usb.sh`
- `restore_new_host.sh`
- `verify_clone_state.sh`
- `sync_mesh_enroll.sh`
- `bootstrap_eidos_machine.sh`
- `post_boot_onboarding.sh`
- `snapshot_eidos_state.sh`
- `refresh_usb_payload.sh`
- `first_boot_wizard.sh`

## USB Build Notes

- `build_dual_live_usb.sh` now defaults `LIVE_MULTI` to `ext4` for stronger Ubuntu casper compatibility.
- `--live-fs exfat` is supported for portability, but can fail ISO discovery on some firmware/initramfs combinations.
- `--print-grub-config` renders boot config without requiring root/device access (used by automated tests).
- GRUB boot entries include ISO path fallbacks (`/ubuntu-live.iso` and `/live/ubuntu-live.iso`).

## Conventions

- All scripts are `bash` and fail fast (`set -euo pipefail`).
- Scripts avoid destructive behavior unless explicitly pointed at a target.
- Manifests are written in deterministic, machine-readable form.
