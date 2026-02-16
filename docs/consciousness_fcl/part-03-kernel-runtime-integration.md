# Part 03: Kernel Runtime Integration

## Goal

Integrate consciousness execution with daemon heartbeat.

## Tasks

1. Instantiate `ConsciousnessKernel` in daemon runtime.
2. Run `kernel.tick()` each beat after base metrics collection.
3. Add config file support for module enable/disable and cadence.
4. Add safety wrappers and timeout guards for module tick paths.

## Termux Notes

- Avoid assumptions about `/proc` fields beyond currently supported ones.
- Keep disk writes append-only and low volume.

## Acceptance Criteria

- Daemon loop remains stable when consciousness modules are disabled.
- Consciousness events appear in standard event bus files.
