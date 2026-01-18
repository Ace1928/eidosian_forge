"""
Runtime‑controlled stellar collapse simulation for Stratum.

This module runs the Stratum simulation for a specified number of
seconds, generating snapshots at fixed wall‑time intervals.  It
extends the basic ``stellar_collapse`` scenario by running until
``runtime_seconds`` elapse, rather than a fixed number of ticks.
Snapshots are saved every ``snapshot_interval`` seconds and bundled
into a zip archive at the end of the run.  The simulation uses
periodic boundaries, a moderate grid size, and high initial thermal
energy to encourage interesting high‑energy interactions (fusion,
decay, degeneracy and black hole formation).

To run this scenario from within the project root:

```
python3 -m stratum.scenarios.stellar_collapse_runtime --runtime 30
```

This will run for 30 seconds of wall time, save a snapshot every
second in ``./stellar_runtime_outputs``, and create a zip archive
``snapshots.zip`` for convenient download.
"""

from __future__ import annotations

import os
import time
import json
import zipfile
from typing import List, Optional

import numpy as np
import matplotlib

# Use non‑interactive backend for headless image generation.
matplotlib.use("Agg")  # type: ignore
import matplotlib.pyplot as plt

from ..core.config import EngineConfig
from ..core.fabric import Fabric
from ..core.ledger import Ledger
from ..core.metronome import Metronome
from ..core.quanta import Quanta
from ..core.registry import SpeciesRegistry
from ..domains.materials.fundamentals import MaterialsFundamentals


def run_stellar_collapse_runtime(
    grid_size: int = 32,
    runtime_seconds: float = 30.0,
    microticks_per_tick: int = 500,
    snapshot_interval: float = 1.0,
    output_dir: str = "./stellar_runtime_outputs",
    *,
    lod_factor: float = 1.0,
    downsample: int = 1,
) -> None:
    """Run the stellar collapse simulation for a given wall‑clock duration.

    This function initialises a Stratum simulation and executes
    ticks until ``runtime_seconds`` have elapsed.  Snapshots of the
    density and temperature fields are saved every
    ``snapshot_interval`` seconds.  When the run completes, all
    snapshot images are packaged into a zip file for convenience.

    Parameters
    ----------
    grid_size : int
        Width and height of the simulation grid.  Should be a small
        integer (e.g. 32 or 64) to permit real‑time performance.
    runtime_seconds : float
        Duration of the simulation in wall‑clock seconds.  The
        simulation continues stepping until this time elapses.
    microticks_per_tick : int
        Number of microticks allocated each tick.  Larger values
        allow more local relaxation and event resolution, but reduce
        the number of ticks achievable per second.  Tune this value
        based on grid size to balance fidelity and performance.
    snapshot_interval : float
        Interval in seconds between successive snapshots.  At least
        one snapshot will be saved at the end of the simulation.
    output_dir : str
        Directory to save snapshot images and the final zip archive.
    lod_factor : float, optional
        Level‑of‑detail scaling factor in (0,1].  Lower values
        reduce the number of mixture species tracked, the microtick
        budget per region and the maximum number of active regions.
        This enables simulations on larger grids (e.g. “galaxy”
        scales) to run efficiently at coarser fidelity.
    downsample : int, optional
        Downsample factor for snapshots.  If greater than one,
        snapshots of the density and temperature fields are averaged
        over ``downsample x downsample`` blocks.  The grid size
        should be divisible by this factor or the remainder cells
        will be ignored in the snapshot.  This does not affect the
        underlying simulation state.

    Returns
    -------
    None
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    # Determine LOD‑dependent configuration values.  Lower LOD reduces
    # mixture complexity, microticks per region and the number of
    # concurrently active regions.  These scalings help simulations on
    # larger grids remain computationally feasible while trading off
    # fine detail.
    # mixture_top_k cannot be less than 1
    base_top_k = 8  # default top‑K in high fidelity mode
    mixture_top_k = max(1, int(base_top_k * lod_factor))
    base_active_regions = 3 * grid_size * grid_size // 4
    active_region_max = max(1, int(base_active_regions * lod_factor))
    # microtick cap scaled by LOD but ensure at least 1
    base_microtick_cap = 10
    microtick_cap_per_region = max(1, int(base_microtick_cap * lod_factor))
    # Configure the simulation
    cfg = EngineConfig(
        grid_w=grid_size,
        grid_h=grid_size,
        entropy_mode=True,
        replay_mode=False,
        boundary="PERIODIC",
        active_region_max=active_region_max,
        microtick_cap_per_region=microtick_cap_per_region,
        tick_budget_ms=200.0,
        Z_fuse_min=1.2,
        Z_deg_min=2.5,
        Z_bh_min=4.5,
        Z_abs_max=6.5,
        Z_star_flip=2.5,
        gravity_strength=0.1,
        mixture_top_k=mixture_top_k,
    )
    # Define HE property names for the registry
    he_props = [
        "HE/rho_max",
        "HE/chi",
        "HE/eta",
        "HE/opacity",
        "HE/kappa_t",
        "HE/kappa_r",
        "HE/beta",
        "HE/nu",
        "HE/lambda",
    ]
    le_props: List[str] = []
    registry_path = os.path.join(output_dir, "species_registry.json")
    registry = SpeciesRegistry(registry_path, he_props, le_props)
    materials = MaterialsFundamentals(registry, cfg)
    fabric = Fabric(cfg)
    ledger = Ledger(fabric, cfg)
    quanta = Quanta(fabric, ledger, registry, materials, cfg)
    # Initialise the grid: fill with stellar gas and high thermal energy
    stellar_id = materials.stellar_species.id
    fabric.rho.fill(1.0)
    fabric.px.fill(0.0)
    fabric.py.fill(0.0)
    fabric.E_heat.fill(5.0)
    fabric.E_rad.fill(0.0)
    for i in range(cfg.grid_w):
        for j in range(cfg.grid_h):
            mix = fabric.mixtures[i][j]
            mix.species_ids = [stellar_id]
            mix.masses = [1.0]
    # Snapshot helper
    def save_snapshot(idx: int, tick: int) -> str:
        """Save a snapshot of the current density and temperature fields.

        If ``downsample > 1``, the density and temperature arrays are
        averaged over ``downsample`` blocks to produce a coarse view.
        This does not modify the underlying simulation state.  When
        downsampling, only the leading dimensions divisible by
        ``downsample`` are used; remainder cells are ignored.
        """
        rho = fabric.rho.copy()
        T_field = np.divide(fabric.E_heat, np.maximum(fabric.rho, 1e-8))
        # Downsample if requested and factor > 1
        if downsample > 1:
            d = downsample
            W, H = rho.shape
            Wc = (W // d) * d
            Hc = (H // d) * d
            # reshape and mean along blocks
            rho = rho[:Wc, :Hc].reshape(Wc // d, d, Hc // d, d).mean(axis=(1, 3))
            T_field = T_field[:Wc, :Hc].reshape(Wc // d, d, Hc // d, d).mean(axis=(1, 3))
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        im0 = axes[0].imshow(rho.T, origin="lower", cmap="inferno")
        axes[0].set_title(f"Density (t={tick})")
        fig.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(T_field.T, origin="lower", cmap="magma")
        axes[1].set_title(f"Temperature (t={tick})")
        fig.colorbar(im1, ax=axes[1])
        plt.tight_layout()
        fname = os.path.join(output_dir, f"snapshot_{idx:03d}.png")
        fig.savefig(fname)
        plt.close(fig)
        return fname
    # Main loop: run until time elapsed
    start_time = time.monotonic()
    next_snap_time = start_time
    snapshot_count = 0
    tick = 0
    # initial snapshot at t=0
    save_snapshot(snapshot_count, tick)
    snapshot_count += 1
    next_snap_time += snapshot_interval
    while time.monotonic() - start_time < runtime_seconds:
        tick += 1
        quanta.step(tick, microticks_per_tick)
        # snapshot based on wall time
        now = time.monotonic()
        if now >= next_snap_time:
            save_snapshot(snapshot_count, tick)
            snapshot_count += 1
            # schedule next snapshot
            next_snap_time += snapshot_interval
        # ensure CPU yields occasionally
        # note: heavy microtick computations may block; we deliberately do not sleep
    # Ensure final snapshot at end of runtime if not recently taken
    if snapshot_count == 0 or time.monotonic() >= next_snap_time:
        save_snapshot(snapshot_count, tick)
        snapshot_count += 1
    # Save species registry
    registry.save()
    # Package snapshots into zip
    zip_path = os.path.join(output_dir, "snapshots.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for idx in range(snapshot_count):
            fname = os.path.join(output_dir, f"snapshot_{idx:03d}.png")
            if os.path.exists(fname):
                zf.write(fname, os.path.basename(fname))
    print(f"Run complete: {snapshot_count} snapshots, {tick} ticks executed.")
    print(f"Snapshots zipped at {zip_path}")


def main():
    """Entry point for the runtime-controlled stellar collapse scenario."""
    import argparse

    parser = argparse.ArgumentParser(description="Run stellar collapse simulation for a fixed runtime.")
    parser.add_argument("--grid", type=int, default=32, help="Grid size (width=height)")
    parser.add_argument("--runtime", type=float, default=30.0, help="Simulation runtime in seconds")
    parser.add_argument("--microticks", type=int, default=500, help="Microticks per tick")
    parser.add_argument("--snapshot", type=float, default=1.0, help="Snapshot interval in seconds")
    parser.add_argument("--output", type=str, default="./stellar_runtime_outputs", help="Output directory")
    args = parser.parse_args()
    run_stellar_collapse_runtime(
        grid_size=args.grid,
        runtime_seconds=args.runtime,
        microticks_per_tick=args.microticks,
        snapshot_interval=args.snapshot,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
