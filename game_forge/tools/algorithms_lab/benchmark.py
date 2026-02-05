#!/usr/bin/env python3
"""Benchmark Algorithms Lab components.

Example:
  python game_forge/tools/algorithms_lab/benchmark.py --algorithms all --particles 1024
  python game_forge/tools/algorithms_lab/benchmark.py --particles 256 --steps 50 --output artifacts/algorithms_lab.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, List

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Algorithms Lab benchmark suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algorithms",
        default="all",
        help="Comma-separated list or 'all'",
    )
    parser.add_argument("--particles", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--fmm-levels", type=int, default=4)
    parser.add_argument("--force-types", type=int, default=6)
    parser.add_argument("--force-multi", action="store_true")
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
        "--output",
        type=Path,
        default=None,
        help="Write benchmark results to JSON",
    )
    return parser.parse_args()


def timer(label: str, fn: Callable[[], None]) -> float:
    start = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - start
    print(f"{label:20s} {elapsed:.4f}s")
    return elapsed


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)

    domain = Domain(
        mins=np.array([0.0, 0.0], dtype=np.float32),
        maxs=np.array([1.0, 1.0], dtype=np.float32),
        wrap=WrapMode.WRAP,
    )
    base_positions = np.random.rand(args.particles, 2).astype(np.float32)
    base_velocities = (np.random.rand(args.particles, 2).astype(np.float32) - 0.5) * 0.1
    masses = np.ones(args.particles, dtype=np.float32)

    selected = (
        [s.strip() for s in args.algorithms.split(",")]
        if args.algorithms != "all"
        else [
            "grid",
            "neighbor-list",
            "barnes-hut",
            "fmm2d",
            "fmm-ml",
            "forces",
            "sph",
            "pbf",
            "xpbd",
        ]
    )

    results: Dict[str, float] = {}

    if "grid" in selected:
        grid = UniformGrid(domain, cell_size=0.05)

        def run_grid() -> None:
            for _ in range(args.steps):
                grid.neighbor_pairs(base_positions, radius=0.05)

        results["grid"] = timer("grid", run_grid)

    if "neighbor-list" in selected:
        nlist = NeighborList(domain, cutoff=0.05, skin=0.01)

        def run_nlist() -> None:
            for _ in range(args.steps):
                nlist.get(base_positions)

        results["neighbor-list"] = timer("neighbor-list", run_nlist)

    if "barnes-hut" in selected:
        tree = BarnesHutTree(domain)

        def run_bh() -> None:
            positions = base_positions.copy()
            velocities = base_velocities.copy()
            for _ in range(args.steps):
                acc = tree.compute_acceleration(
                    positions, masses, theta=0.6, backend=args.bh_backend
                )
                velocities = velocities + acc * args.dt
                positions = domain.wrap_positions(positions + velocities * args.dt)

        results["barnes-hut"] = timer("barnes-hut", run_bh)

    if "fmm2d" in selected:
        fmm = FMM2D(domain, cell_size=0.1)

        def run_fmm() -> None:
            positions = base_positions.copy()
            velocities = base_velocities.copy()
            for _ in range(args.steps):
                acc = fmm.compute_acceleration(positions, masses)
                velocities = velocities + acc * args.dt
                positions = domain.wrap_positions(positions + velocities * args.dt)

        results["fmm2d"] = timer("fmm2d", run_fmm)

    if "fmm-ml" in selected:
        fmm = MultiLevelFMM(domain, levels=args.fmm_levels)

        def run_fmm_ml() -> None:
            positions = base_positions.copy()
            velocities = base_velocities.copy()
            for _ in range(args.steps):
                acc = fmm.compute_acceleration(positions, masses)
                velocities = velocities + acc * args.dt
                positions = domain.wrap_positions(positions + velocities * args.dt)

        results["fmm-ml"] = timer("fmm-ml", run_fmm_ml)

    if "forces" in selected:
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

        def run_forces() -> None:
            positions = base_positions.copy()
            velocities = base_velocities.copy()
            for _ in range(args.steps):
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
                    masses=masses,
                )
                velocities = velocities + acc * args.dt
                positions = domain.wrap_positions(positions + velocities * args.dt)

        results["forces"] = timer("forces", run_forces)

    if "sph" in selected:
        solver = SPHSolver(domain, h=0.06, dt=args.dt, neighbor_backend=args.neighbor_backend)
        state = SPHState(positions=base_positions, velocities=base_velocities, masses=masses)

        def run_sph() -> None:
            nonlocal state
            for _ in range(args.steps):
                state = solver.step(state)

        results["sph"] = timer("sph", run_sph)

    if "pbf" in selected:
        solver = PBFSolver(domain, h=0.06, dt=args.dt, neighbor_backend=args.neighbor_backend)
        state = PBFState(positions=base_positions, velocities=base_velocities, masses=masses)

        def run_pbf() -> None:
            nonlocal state
            for _ in range(args.steps):
                state = solver.step(state)

        results["pbf"] = timer("pbf", run_pbf)

    if "xpbd" in selected:
        solver = XPBFSolver(
            domain,
            h=0.06,
            dt=args.dt,
            compliance=0.001,
            neighbor_backend=args.neighbor_backend,
        )
        state = XPBFState(positions=base_positions, velocities=base_velocities, masses=masses)

        def run_xpbd() -> None:
            nonlocal state
            for _ in range(args.steps):
                state = solver.step(state)

        results["xpbd"] = timer("xpbd", run_xpbd)

    print("INFO benchmark complete")

    if args.output is not None:
        payload = {
            "parameters": {
                "algorithms": args.algorithms,
                "particles": args.particles,
                "steps": args.steps,
                "seed": args.seed,
                "dt": args.dt,
                "fmm_levels": args.fmm_levels,
                "force_types": args.force_types,
                "force_multi": args.force_multi,
                "bh_backend": args.bh_backend,
                "neighbor_backend": args.neighbor_backend,
            },
            "results_seconds": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"INFO wrote results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
