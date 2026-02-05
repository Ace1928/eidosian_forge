#!/usr/bin/env python3
"""Visual and headless demos for Algorithms Lab.

Example:
  python game_forge/tools/algorithms_lab/demo.py --algorithm sph --visual
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import numpy as np

from algorithms_lab.backends import HAS_NUMBA
from algorithms_lab.barnes_hut import BarnesHutTree
from algorithms_lab.core import Domain, WrapMode
from algorithms_lab.forces import ForceRegistry, accumulate_from_registry
from algorithms_lab.graph import build_neighbor_graph
from algorithms_lab.fmm2d import FMM2D
from algorithms_lab.fmm_multilevel import MultiLevelFMM
from algorithms_lab.neighbor_list import NeighborList
from algorithms_lab.spatial_hash import UniformGrid
from algorithms_lab.sph import SPHState, SPHSolver
from algorithms_lab.pbf import PBFState, PBFSolver
from algorithms_lab.xpbd import XPBFState, XPBFSolver

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
        choices=[
            "grid",
            "neighbor-list",
            "barnes-hut",
            "fmm2d",
            "fmm-ml",
            "forces",
            "sph",
            "pbf",
            "xpbd",
        ],
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
    parser.add_argument("--fmm-levels", type=int, default=4)
    parser.add_argument("--force-types", type=int, default=6)
    parser.add_argument("--force-multi", action="store_true", help="Enable multiple force families")
    parser.add_argument(
        "--bh-backend",
        choices=["auto", "numpy", "numba"],
        default="auto",
        help="Backend for Barnes-Hut traversal",
    )
    parser.add_argument(
        "--neighbor-backend",
        choices=["auto", "numpy", "numba"],
        default="auto",
        help="Neighbor backend for SPH/PBF",
    )
    parser.add_argument(
        "--style",
        choices=["classic", "modern"],
        default="modern",
        help="Visual style",
    )
    parser.add_argument("--point-size", type=int, default=3)
    return parser.parse_args()


def jittered_grid(count: int) -> np.ndarray:
    side = int(np.ceil(np.sqrt(count)))
    xs, ys = np.meshgrid(np.linspace(0.05, 0.95, side), np.linspace(0.05, 0.95, side))
    coords = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=-1)
    coords = coords[:count]
    coords += (np.random.rand(*coords.shape) - 0.5) * 0.02
    return coords.astype(np.float32)


@dataclass
class VisualTheme:
    background_top: Tuple[int, int, int]
    background_bottom: Tuple[int, int, int]
    particle: Tuple[int, int, int]
    glow: Tuple[int, int, int]
    panel_bg: Tuple[int, int, int, int]
    panel_border: Tuple[int, int, int]
    text: Tuple[int, int, int]


def _blend_color(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


def build_background(size: Tuple[int, int], theme: VisualTheme) -> pygame.Surface:
    width, height = size
    surface = pygame.Surface(size)
    for y in range(height):
        t = y / max(1, height - 1)
        color = _blend_color(theme.background_top, theme.background_bottom, t)
        pygame.draw.line(surface, color, (0, y), (width, y))
    rng = np.random.default_rng(42)
    for _ in range(120):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        shade = int(rng.integers(180, 255))
        surface.set_at((x, y), (shade, shade, shade))
    return surface


def render_particles_modern(
    screen: pygame.Surface,
    background: pygame.Surface,
    positions: np.ndarray,
    theme: VisualTheme,
    font: pygame.font.Font,
    hud_lines: Iterable[str],
    point_size: int,
) -> None:
    screen.blit(background, (0, 0))
    width, height = screen.get_size()
    for x, y in positions:
        px = int(x * width)
        py = int(y * height)
        for radius, alpha in ((point_size + 6, 30), (point_size + 2, 80), (point_size, 220)):
            glow = pygame.Surface((radius * 2 + 2, radius * 2 + 2), pygame.SRCALPHA)
            pygame.draw.circle(glow, (*theme.glow, alpha), (radius + 1, radius + 1), radius)
            screen.blit(glow, (px - radius - 1, py - radius - 1), special_flags=pygame.BLEND_ALPHA_SDL2)
        pygame.draw.circle(screen, theme.particle, (px, py), point_size)
    panel = pygame.Surface((260, 130), pygame.SRCALPHA)
    panel.fill(theme.panel_bg)
    pygame.draw.rect(panel, theme.panel_border, panel.get_rect(), width=1, border_radius=8)
    y = 10
    for line in hud_lines:
        text = font.render(line, True, theme.text)
        panel.blit(text, (12, y))
        y += 22
    screen.blit(panel, (12, 12))


def render_particles_classic(
    screen: pygame.Surface, positions: np.ndarray, color: Tuple[int, int, int], point_size: int
) -> None:
    screen.fill((10, 12, 18))
    width, height = screen.get_size()
    for x, y in positions:
        px = int(x * width)
        py = int(y * height)
        pygame.draw.circle(screen, color, (px, py), point_size)


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
            acc = tree.compute_acceleration(
                positions, masses, theta=0.6, backend=args.bh_backend
            )
            velocities = velocities + acc * args.dt
            positions = domain.wrap_positions(positions + velocities * args.dt)

    elif args.algorithm == "fmm2d":
        fmm = FMM2D(domain, cell_size=0.1)

        def step_fn(_: int) -> None:
            nonlocal positions, velocities
            acc = fmm.compute_acceleration(positions, masses)
            velocities = velocities + acc * args.dt
            positions = domain.wrap_positions(positions + velocities * args.dt)

    elif args.algorithm == "fmm-ml":
        fmm = MultiLevelFMM(domain, levels=args.fmm_levels)

        def step_fn(_: int) -> None:
            nonlocal positions, velocities
            acc = fmm.compute_acceleration(positions, masses)
            velocities = velocities + acc * args.dt
            positions = domain.wrap_positions(positions + velocities * args.dt)

    elif args.algorithm == "forces":
        registry = ForceRegistry(num_types=args.force_types)
        registry.randomize_all()
        if args.force_multi:
            for name in ("Yukawa", "Lennard-Jones", "Morse Bond", "Gravity"):
                force = registry.get_force(name)
                if force is not None:
                    force.enabled = True
                    force.randomize_matrix()
        type_ids = np.random.randint(0, args.force_types, size=args.particles, dtype=np.int32)
        graph_backend = "numba" if HAS_NUMBA and args.neighbor_backend in ("auto", "numba") else "numpy"

        def step_fn(_: int) -> None:
            nonlocal positions, velocities
            graph = build_neighbor_graph(
                positions,
                radius=registry.get_max_radius(),
                domain=domain,
                method="grid",
                backend=graph_backend,
            )
            acc = accumulate_from_registry(
                positions,
                type_ids,
                graph.rows,
                graph.cols,
                registry,
                domain,
            )
            velocities = velocities + acc * args.dt
            positions = domain.wrap_positions(positions + velocities * args.dt)

    elif args.algorithm == "sph":
        solver = SPHSolver(
            domain, h=0.06, dt=args.dt, gravity=0.0, neighbor_backend=args.neighbor_backend
        )
        state = SPHState(positions=positions, velocities=velocities, masses=masses)

        def step_fn(_: int) -> None:
            nonlocal state
            state = solver.step(state)

    elif args.algorithm == "pbf":
        solver = PBFSolver(
            domain, h=0.06, dt=args.dt, gravity=0.0, neighbor_backend=args.neighbor_backend
        )
        state = PBFState(positions=positions, velocities=velocities, masses=masses)

        def step_fn(_: int) -> None:
            nonlocal state
            state = solver.step(state)

    elif args.algorithm == "xpbd":
        solver = XPBFSolver(
            domain,
            h=0.06,
            dt=args.dt,
            gravity=0.0,
            compliance=0.001,
            neighbor_backend=args.neighbor_backend,
        )
        state = XPBFState(positions=positions, velocities=velocities, masses=masses)

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
    theme = VisualTheme(
        background_top=(10, 12, 25),
        background_bottom=(5, 5, 8),
        particle=(210, 230, 255),
        glow=(120, 160, 255),
        panel_bg=(10, 14, 20, 170),
        panel_border=(70, 90, 120),
        text=(220, 230, 240),
    )
    font = pygame.font.SysFont("Fira Sans", 16)
    background = build_background(screen.get_size(), theme)

    running = True
    step = 0
    while running and step < args.steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        step_fn(step)
        if args.algorithm in ("sph", "pbf", "xpbd"):
            pos = state.positions
        else:
            pos = positions
        if args.style == "modern":
            hud = [
                f"Algorithm: {args.algorithm}",
                f"Particles: {args.particles}",
                f"Step: {step}",
                f"Backend: {args.neighbor_backend}",
            ]
            render_particles_modern(
                screen,
                background,
                pos,
                theme,
                font,
                hud,
                args.point_size,
            )
        else:
            render_particles_classic(screen, pos, (200, 200, 255), args.point_size)
        pygame.display.flip()
        clock.tick(60)
        step += 1

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
