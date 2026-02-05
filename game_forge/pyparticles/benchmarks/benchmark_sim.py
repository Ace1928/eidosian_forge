from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
import sys
import time
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from pyparticles.core.types import SimulationConfig
from pyparticles.physics.engine import PhysicsEngine


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the PyParticles simulation step",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--particles", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None, help="Write JSON summary output")
    parser.add_argument("--profile", action="store_true", default=True, help="Enable cProfile output")
    parser.add_argument("--no-profile", dest="profile", action="store_false", help="Disable cProfile output")
    return parser.parse_args(argv)


def benchmark(args: argparse.Namespace) -> dict[str, float]:
    n_particles = args.particles
    steps = args.steps
    dt = args.dt
    print(f"Benchmarking with N={n_particles} particles...")
    cfg = SimulationConfig.small_world() if hasattr(SimulationConfig, "small_world") else SimulationConfig.default()
    cfg.num_particles = n_particles
    engine = PhysicsEngine(cfg)

    if args.warmup > 0:
        print("Warming up JIT...")
        for _ in range(args.warmup):
            engine.update(dt)
        print("Warmup complete.")

    start_time = time.perf_counter()
    for _ in range(steps):
        engine.update(dt)
    duration = time.perf_counter() - start_time

    fps = steps / duration if duration > 0 else 0.0
    print(f"Processed {steps} frames in {duration:.4f}s")
    print(f"Simulation FPS: {fps:.2f}")
    return {
        "particles": n_particles,
        "steps": steps,
        "dt": dt,
        "warmup": args.warmup,
        "elapsed_seconds": duration,
        "fps": fps,
    }


def write_output(path: Path, payload: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        payload = benchmark(args)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumtime")
        stats.print_stats(20)
    else:
        payload = benchmark(args)

    if args.output is not None:
        write_output(args.output, payload)
        print(f"INFO wrote results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
