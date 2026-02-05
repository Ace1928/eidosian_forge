#!/usr/bin/env python3
"""Profile Algorithms Lab components with cProfile.

Example:
  python game_forge/tools/algorithms_lab/profile.py --algorithm barnes-hut
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
from pathlib import Path

import numpy as np

from algorithms_lab.barnes_hut import BarnesHutTree
from algorithms_lab.core import Domain, WrapMode
from algorithms_lab.fmm2d import FMM2D
from algorithms_lab.spatial_hash import UniformGrid
from algorithms_lab.neighbor_list import NeighborList
from algorithms_lab.sph import SPHState, SPHSolver
from algorithms_lab.pbf import PBFState, PBFSolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Algorithms Lab profiler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algorithm",
        choices=["grid", "neighbor-list", "barnes-hut", "fmm2d", "sph", "pbf"],
        default="barnes-hut",
    )
    parser.add_argument("--particles", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=Path("/tmp/algorithms_lab.prof"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    np.random.seed(7)

    domain = Domain(
        mins=np.array([0.0, 0.0], dtype=np.float32),
        maxs=np.array([1.0, 1.0], dtype=np.float32),
        wrap=WrapMode.WRAP,
    )
    positions = np.random.rand(args.particles, 2).astype(np.float32)
    velocities = (np.random.rand(args.particles, 2).astype(np.float32) - 0.5) * 0.1
    masses = np.ones(args.particles, dtype=np.float32)

    if args.algorithm == "grid":
        grid = UniformGrid(domain, cell_size=0.05)

        def run() -> None:
            for _ in range(args.steps):
                grid.neighbor_pairs(positions, radius=0.05)

    elif args.algorithm == "neighbor-list":
        nlist = NeighborList(domain, cutoff=0.05, skin=0.01)

        def run() -> None:
            for _ in range(args.steps):
                nlist.get(positions)

    elif args.algorithm == "barnes-hut":
        tree = BarnesHutTree(domain)

        def run() -> None:
            nonlocal positions, velocities
            for _ in range(args.steps):
                acc = tree.compute_acceleration(positions, masses, theta=0.6)
                velocities = velocities + acc * args.dt
                positions = domain.wrap_positions(positions + velocities * args.dt)

    elif args.algorithm == "fmm2d":
        fmm = FMM2D(domain, cell_size=0.1)

        def run() -> None:
            nonlocal positions, velocities
            for _ in range(args.steps):
                acc = fmm.compute_acceleration(positions, masses)
                velocities = velocities + acc * args.dt
                positions = domain.wrap_positions(positions + velocities * args.dt)

    elif args.algorithm == "sph":
        solver = SPHSolver(domain, h=0.06, dt=args.dt)
        state = SPHState(positions=positions, velocities=velocities, masses=masses)

        def run() -> None:
            nonlocal state
            for _ in range(args.steps):
                state = solver.step(state)

    elif args.algorithm == "pbf":
        solver = PBFSolver(domain, h=0.06, dt=args.dt)
        state = PBFState(positions=positions, velocities=velocities, masses=masses)

        def run() -> None:
            nonlocal state
            for _ in range(args.steps):
                state = solver.step(state)

    else:
        raise ValueError("Unknown algorithm")

    profiler = cProfile.Profile()
    profiler.enable()
    run()
    profiler.disable()
    profiler.dump_stats(str(args.output))
    stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")
    stats.print_stats(20)
    print(f"INFO profile saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
