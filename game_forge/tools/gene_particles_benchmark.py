#!/usr/bin/env python3
"""Benchmark Gene Particles core simulation steps.

Example:
  python game_forge/tools/gene_particles_benchmark.py --steps 50 --cell-types 3 --particles 200
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
    for ct in automata.type_manager.cellular_types:
        automata.apply_clustering(ct)
    if automata.config.reproduction_mode in (
        ReproductionMode.MANAGER,
        ReproductionMode.HYBRID,
    ):
        automata.type_manager.reproduce()
    automata.type_manager.remove_dead_in_all_types()
    automata.update_species_count()


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Gene Particles steps")
    parser.add_argument("--steps", type=int, default=20, help="Number of steps to run")
    parser.add_argument("--cell-types", type=int, default=3, help="Number of cell types")
    parser.add_argument("--particles", type=int, default=200, help="Particles per type")
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
    parser.add_argument(
        "--gene-interpreter",
        action="store_true",
        default=False,
        help="Enable genetic interpreter",
    )
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

    np.random.seed(args.seed)
    pygame.init()
    automata = build_automata(args)

    # Warm-up
    step_once(automata)

    start = time.perf_counter()
    for _ in range(args.steps):
        step_once(automata)
    elapsed = time.perf_counter() - start

    per_step_ms = (elapsed / max(1, args.steps)) * 1000.0
    print(f"INFO steps={args.steps} total_s={elapsed:.4f} ms_per_step={per_step_ms:.3f}")

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
