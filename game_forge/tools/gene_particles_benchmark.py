#!/usr/bin/env python3
"""Benchmark Gene Particles core simulation steps.

Example:
  python game_forge/tools/gene_particles_benchmark.py --steps 1000 --cell-types 8 --particles 500
  python game_forge/tools/gene_particles_benchmark.py --gene-interpreter --reproduction-mode hybrid
"""

import argparse
import os
import sys
import time
import warnings

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"pkg_resources",
)

import numpy as np
import pygame

from game_forge.src.gene_particles.gp_automata import CellularAutomata
from game_forge.src.gene_particles.gp_config import ReproductionMode, SimulationConfig


def build_automata(args: argparse.Namespace) -> CellularAutomata:
    config = SimulationConfig()
    config.n_cell_types = args.cell_types
    config.particles_per_type = args.particles
    config.max_particles_per_type = max(config.max_particles_per_type, args.particles)
    config.mass_based_fraction = args.mass_fraction
    config.max_frames = 0
    config.spatial_dimensions = args.dimensions
    config.world_depth = args.world_depth
    config.use_gene_interpreter = args.gene_interpreter
    config.reproduction_mode = ReproductionMode(args.reproduction_mode)
    config.use_force_registry = args.force_registry
    config.force_registry_min_particles = args.force_registry_min_particles
    config.force_registry_family_scale["yukawa"] = args.force_yukawa
    config.force_registry_family_scale["lennard_jones"] = args.force_lj
    config.force_registry_family_scale["morse"] = args.force_morse
    config.use_morton_ordering = args.morton
    config.morton_min_particles = args.morton_min_particles
    config.morton_cell_scale = args.morton_cell_scale
    return CellularAutomata(
        config,
        fullscreen=False,
        screen_size=(args.width, args.height),
    )


def step_once(automata: CellularAutomata) -> None:
    automata.frame_count += 1
    automata.config.advance_environment(automata.frame_count)
    automata.apply_all_interactions()
    if (
        automata.interpreter_enabled
        and automata.frame_count % automata.config.gene_interpreter_interval == 0
    ):
        automata.apply_gene_interpreter()
    automata.apply_clustering_all()
    if automata.config.reproduction_mode in (
        ReproductionMode.MANAGER,
        ReproductionMode.HYBRID,
    ):
        automata.type_manager.reproduce()
    automata.type_manager.remove_dead_in_all_types()
    automata.update_species_count()


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Gene Particles steps")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to run")
    parser.add_argument("--cell-types", type=int, default=8, help="Number of cell types")
    parser.add_argument("--particles", type=int, default=500, help="Particles per type")
    parser.add_argument("--mass-fraction", type=float, default=0.5, help="Mass-based fraction")
    parser.add_argument("--width", type=int, default=800, help="Window width")
    parser.add_argument("--height", type=int, default=600, help="Window height")
    parser.add_argument(
        "--dimensions",
        type=int,
        choices=[2, 3],
        default=3,
        help="Simulation dimensionality",
    )
    parser.add_argument(
        "--world-depth",
        type=float,
        default=None,
        help="Depth of the simulation volume (3D only)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps")
    parser.add_argument("--output", type=str, default="", help="Write metrics to JSON")
    parser.add_argument(
        "--gene-interpreter",
        action="store_true",
        default=False,
        help="Enable genetic interpreter",
    )
    parser.add_argument(
        "--force-registry",
        action="store_true",
        default=False,
        help="Enable force registry interactions",
    )
    parser.add_argument(
        "--force-registry-min-particles",
        type=int,
        default=512,
        help="Minimum particles to enable force registry",
    )
    parser.add_argument("--force-yukawa", type=float, default=0.0)
    parser.add_argument("--force-lj", type=float, default=0.0)
    parser.add_argument("--force-morse", type=float, default=0.0)
    parser.add_argument("--morton", action="store_true", default=False)
    parser.add_argument("--morton-min-particles", type=int, default=2048)
    parser.add_argument("--morton-cell-scale", type=float, default=1.0)
    parser.add_argument(
        "--reproduction-mode",
        type=str,
        choices=["manager", "genes", "hybrid"],
        default="manager",
        help="Reproduction pipeline to benchmark",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Use SDL dummy video driver",
    )
    parser.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help="Use real video driver",
    )
    args = parser.parse_args()

    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    np.random.seed(args.seed)
    pygame.init()
    automata = build_automata(args)

    for _ in range(max(0, args.warmup)):
        step_once(automata)

    step_times = []
    start = time.perf_counter()
    for _ in range(args.steps):
        step_start = time.perf_counter()
        step_once(automata)
        step_times.append(time.perf_counter() - step_start)
    elapsed = time.perf_counter() - start

    step_arr = np.array(step_times, dtype=np.float64)
    per_step_ms = float(np.mean(step_arr) * 1000.0)
    median_ms = float(np.median(step_arr) * 1000.0)
    p95_ms = float(np.percentile(step_arr, 95) * 1000.0)
    p99_ms = float(np.percentile(step_arr, 99) * 1000.0)
    min_ms = float(np.min(step_arr) * 1000.0)
    max_ms = float(np.max(step_arr) * 1000.0)
    steps_per_s = float(args.steps / max(elapsed, 1e-9))

    summary = {
        "steps": args.steps,
        "total_s": elapsed,
        "mean_ms": per_step_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "steps_per_s": steps_per_s,
        "cell_types": args.cell_types,
        "particles": args.particles,
        "dimensions": args.dimensions,
        "force_registry": args.force_registry,
        "force_yukawa": args.force_yukawa,
        "force_lj": args.force_lj,
        "force_morse": args.force_morse,
        "morton": args.morton,
    }

    print(
        "INFO "
        f"steps={args.steps} total_s={elapsed:.4f} "
        f"mean_ms={per_step_ms:.3f} median_ms={median_ms:.3f} "
        f"p95_ms={p95_ms:.3f} p99_ms={p99_ms:.3f} "
        f"min_ms={min_ms:.3f} max_ms={max_ms:.3f} "
        f"steps_per_s={steps_per_s:.2f}"
    )

    if args.output:
        import json

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"INFO wrote metrics to {args.output}")

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
