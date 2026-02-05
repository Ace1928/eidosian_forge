#!/usr/bin/env python3
"""Visual and headless demos for Algorithms Lab.

Example:
  python game_forge/tools/algorithms_lab/demo.py --algorithm sph --visual
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable, Tuple

import numpy as np

from algorithms_lab.barnes_hut import BarnesHutTree
from algorithms_lab.core import Domain, WrapMode
from algorithms_lab.fmm2d import FMM2D
from algorithms_lab.neighbor_list import NeighborList
from algorithms_lab.spatial_hash import UniformGrid
from algorithms_lab.sph import SPHState, SPHSolver
from algorithms_lab.pbf import PBFState, PBFSolver

try:
    import pygame

    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Algorithms Lab demos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algorithm",
        choices=["grid", "neighbor-list", "barnes-hut", "fmm2d", "sph", "pbf"],
        default="sph",
        help="Which demo to run",
    )
    parser.add_argument("--particles", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--visual", action="store_true", help="Enable pygame demo")
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--dt", type=float, default=0.01)
    return parser.parse_args()


def jittered_grid(count: int) -> np.ndarray:
    side = int(np.ceil(np.sqrt(count)))
    xs, ys = np.meshgrid(np.linspace(0.05, 0.95, side), np.linspace(0.05, 0.95, side))
    coords = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=-1)
    coords = coords[:count]
    coords += (np.random.rand(*coords.shape) - 0.5) * 0.02
    return coords.astype(np.float32)


def render_particles(screen: pygame.Surface, positions: np.ndarray, color: Tuple[int, int, int]) -> None:
    screen.fill((10, 12, 18))
    width, height = screen.get_size()
    for x, y in positions:
        px = int(x * width)
        py = int(y * height)
        pygame.draw.circle(screen, color, (px, py), 3)


def run_headless(step_fn: Callable[[int], None], steps: int) -> None:
    start = time.perf_counter()
    for step in range(steps):
        step_fn(step)
    elapsed = time.perf_counter() - start
    print(f"INFO ran {steps} steps in {elapsed:.3f}s ({steps/elapsed:.1f} steps/s)")


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)

    if args.visual and not HAS_PYGAME:
        print("ERROR pygame is not available; rerun without --visual")
        return 1

    domain = Domain(
        mins=np.array([0.0, 0.0], dtype=np.float32),
        maxs=np.array([1.0, 1.0], dtype=np.float32),
        wrap=WrapMode.WRAP,
    )
    positions = jittered_grid(args.particles)
    velocities = (np.random.rand(args.particles, 2).astype(np.float32) - 0.5) * 0.1
    masses = np.ones(args.particles, dtype=np.float32)

    if args.algorithm == "grid":
        grid = UniformGrid(domain, cell_size=0.05)

        def step_fn(_: int) -> None:
            grid.neighbor_pairs(positions, radius=0.05)

    elif args.algorithm == "neighbor-list":
        nlist = NeighborList(domain, cutoff=0.05, skin=0.01)

        def step_fn(_: int) -> None:
            nlist.get(positions)

    elif args.algorithm == "barnes-hut":
        tree = BarnesHutTree(domain)

        def step_fn(_: int) -> None:
            nonlocal positions, velocities
            acc = tree.compute_acceleration(positions, masses, theta=0.6)
            velocities = velocities + acc * args.dt
            positions = domain.wrap_positions(positions + velocities * args.dt)

    elif args.algorithm == "fmm2d":
        fmm = FMM2D(domain, cell_size=0.1)

        def step_fn(_: int) -> None:
            nonlocal positions, velocities
            acc = fmm.compute_acceleration(positions, masses)
            velocities = velocities + acc * args.dt
            positions = domain.wrap_positions(positions + velocities * args.dt)

    elif args.algorithm == "sph":
        solver = SPHSolver(domain, h=0.06, dt=args.dt, gravity=0.0)
        state = SPHState(positions=positions, velocities=velocities, masses=masses)

        def step_fn(_: int) -> None:
            nonlocal state
            state = solver.step(state)

    elif args.algorithm == "pbf":
        solver = PBFSolver(domain, h=0.06, dt=args.dt, gravity=0.0)
        state = PBFState(positions=positions, velocities=velocities, masses=masses)

        def step_fn(_: int) -> None:
            nonlocal state
            state = solver.step(state)

    else:
        raise ValueError("Unknown algorithm")

    if not args.visual:
        run_headless(step_fn, args.steps)
        return 0

    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption(f"Algorithms Lab Demo: {args.algorithm}")
    clock = pygame.time.Clock()

    running = True
    step = 0
    while running and step < args.steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        step_fn(step)
        if args.algorithm in ("sph", "pbf"):
            pos = state.positions
        else:
            pos = positions
        render_particles(screen, pos, (200, 200, 255))
        pygame.display.flip()
        clock.tick(60)
        step += 1

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
