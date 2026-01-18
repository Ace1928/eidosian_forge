"""
Unified CLI for launching Stratum scenarios.
"""

from __future__ import annotations

import argparse
from typing import Optional

SCENARIOS = {
    "collapse": "Stellar collapse (fixed ticks + snapshots)",
    "runtime": "Stellar collapse (fixed wall-time + snapshots)",
    "screensaver": "Real-time screensaver (Pygame)",
}


def _value_or_default(value: Optional[float], default: float) -> float:
    return default if value is None else value


def _prompt_int(label: str, default: int, min_value: int = 1) -> int:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Please enter an integer.")
            continue
        if value < min_value:
            print(f"Value must be >= {min_value}.")
            continue
        return value


def _prompt_float(label: str, default: float, min_value: Optional[float] = None) -> float:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if min_value is not None and value < min_value:
            print(f"Value must be >= {min_value}.")
            continue
        return value


def _prompt_choice() -> str:
    print("Available scenarios:")
    for key, desc in SCENARIOS.items():
        print(f"  - {key}: {desc}")
    while True:
        raw = input("Choose scenario: ").strip().lower()
        if raw in SCENARIOS:
            return raw
        print("Unknown scenario. Choose one of: " + ", ".join(SCENARIOS.keys()))


def _run_collapse(args: argparse.Namespace, interactive: bool) -> None:
    from .stellar_collapse import run_stellar_collapse
    grid_default = 32
    ticks_default = 500
    micro_default = 500
    output_default = "./stellar_outputs"

    grid_seed = int(_value_or_default(args.grid, grid_default))
    ticks_seed = int(_value_or_default(args.ticks, ticks_default))
    micro_seed = int(_value_or_default(args.microticks, micro_default))
    output_seed = args.output or output_default

    if interactive:
        grid = _prompt_int("Grid size", grid_seed)
        ticks = _prompt_int("Ticks", ticks_seed)
        microticks = _prompt_int("Microticks per tick", micro_seed)
        output = input(f"Output directory [{output_seed}]: ").strip() or output_seed
    else:
        grid = max(1, grid_seed)
        ticks = max(1, ticks_seed)
        microticks = max(1, micro_seed)
        output = output_seed

    run_stellar_collapse(
        grid_size=grid,
        num_ticks=ticks,
        microticks_per_tick=microticks,
        output_dir=output,
    )


def _run_runtime(args: argparse.Namespace, interactive: bool) -> None:
    from .stellar_collapse_runtime import run_stellar_collapse_runtime
    grid_default = 32
    runtime_default = 30.0
    micro_default = 500
    snapshot_default = 1.0
    output_default = "./stellar_runtime_outputs"
    lod_default = 1.0
    downsample_default = 1

    grid_seed = int(_value_or_default(args.grid, grid_default))
    runtime_seed = float(_value_or_default(args.runtime, runtime_default))
    micro_seed = int(_value_or_default(args.microticks, micro_default))
    snapshot_seed = float(_value_or_default(args.snapshot, snapshot_default))
    output_seed = args.output or output_default
    lod_seed = float(_value_or_default(args.lod, lod_default))
    downsample_seed = int(_value_or_default(args.downsample, downsample_default))

    if interactive:
        grid = _prompt_int("Grid size", grid_seed)
        runtime = _prompt_float("Runtime seconds", runtime_seed, min_value=0.1)
        microticks = _prompt_int("Microticks per tick", micro_seed)
        snapshot = _prompt_float("Snapshot interval seconds", snapshot_seed, min_value=0.1)
        output = input(f"Output directory [{output_seed}]: ").strip() or output_seed
        lod = _prompt_float("LOD factor", lod_seed, min_value=0.05)
        downsample = _prompt_int("Downsample factor", downsample_seed)
    else:
        grid = max(1, grid_seed)
        runtime = max(0.1, runtime_seed)
        microticks = max(1, micro_seed)
        snapshot = max(0.1, snapshot_seed)
        output = output_seed
        lod = max(0.05, min(lod_seed, 1.0))
        downsample = max(1, downsample_seed)

    run_stellar_collapse_runtime(
        grid_size=grid,
        runtime_seconds=runtime,
        microticks_per_tick=microticks,
        snapshot_interval=snapshot,
        output_dir=output,
        lod_factor=lod,
        downsample=downsample,
    )


def _run_screensaver(args: argparse.Namespace, interactive: bool) -> None:
    from .stellar_screensaver import run_stellar_screensaver
    grid_default = 0
    micro_default = 50
    scale_default = 1
    lod_default = 1.0
    fps_default = 30
    gravity_default = 0.05

    grid_seed = int(_value_or_default(args.grid, grid_default))
    micro_seed = int(_value_or_default(args.microticks, micro_default))
    scale_seed = int(_value_or_default(args.scale, scale_default))
    lod_seed = float(_value_or_default(args.lod, lod_default))
    fps_seed = int(_value_or_default(args.fps, fps_default))
    gravity_seed = float(_value_or_default(args.gravity, gravity_default))

    if interactive:
        grid = _prompt_int("Grid size (0=auto)", grid_seed, min_value=0)
        microticks = _prompt_int("Microticks per tick", micro_seed)
        scale = _prompt_int("Pixel scale", scale_seed)
        lod = _prompt_float("LOD factor", lod_seed, min_value=0.05)
        fps = _prompt_int("Target FPS", fps_seed)
        gravity = _prompt_float("Gravity strength", gravity_seed, min_value=0.0)
    else:
        grid = max(0, grid_seed)
        microticks = max(1, micro_seed)
        scale = max(1, scale_seed)
        lod = max(0.05, min(lod_seed, 1.0))
        fps = max(5, fps_seed)
        gravity = max(0.0, gravity_seed)

    run_stellar_screensaver(
        grid_size=grid,
        microticks_per_tick=microticks,
        scale=scale,
        lod_factor=lod,
        fps=fps,
        gravity_strength=gravity,
    )


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Stratum unified scenario launcher")
    parser.add_argument("--list", action="store_true", help="List available scenarios")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS.keys()), help="Scenario to run")
    parser.add_argument("--grid", type=int, help="Grid size (width=height)")
    parser.add_argument("--ticks", type=int, help="Number of ticks (collapse scenario)")
    parser.add_argument("--microticks", type=int, help="Microticks per tick")
    parser.add_argument("--runtime", type=float, help="Runtime in seconds (runtime scenario)")
    parser.add_argument("--snapshot", type=float, help="Snapshot interval in seconds")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--scale", type=int, help="Pixel scale (screensaver scenario)")
    parser.add_argument("--lod", type=float, help="Level-of-detail factor")
    parser.add_argument("--fps", type=int, help="Target FPS (screensaver scenario)")
    parser.add_argument("--gravity", type=float, help="Gravity strength (screensaver scenario)")
    parser.add_argument("--downsample", type=int, help="Downsample factor (runtime scenario)")
    args = parser.parse_args(argv)

    if args.list:
        for key, desc in SCENARIOS.items():
            print(f"{key}: {desc}")
        return

    interactive = args.scenario is None
    scenario = _prompt_choice() if interactive else args.scenario

    if scenario == "collapse":
        _run_collapse(args, interactive)
    elif scenario == "runtime":
        _run_runtime(args, interactive)
    elif scenario == "screensaver":
        _run_screensaver(args, interactive)
    else:
        raise SystemExit(f"Unknown scenario: {scenario}")


if __name__ == "__main__":
    main()
