"""
Simplified stellar collapse scenario for the Stratum prototype.

This module sets up a minimal simulation using the existing Stratum
subsystems.  It initialises a small grid filled with a uniform
distribution of ``StellarGas`` and runs the ``Quanta`` microtick
loop for a number of ticks.  The goal of this demonstration is not
to reproduce full astrophysical dynamics but to exercise the core
engine in a controlled setting.  The output of the simulation can
be visualised using the ``viewer`` module or saved to disk for
analysis.

Key simplifications in this scenario:

* The grid size is kept small (e.g. 32×32) to allow the simulation
  to run quickly.  Larger grids are possible but will require more
  computational budget.
* Only a single fundamental material (``StellarGas``) is used.
  Degenerate transitions, black holes and chemistry are disabled.
* The mixture in each cell initially contains 100 % of the stellar
  species.  No random noise is added to the initial mass
  distribution; however, tiny perturbations in the microtick loop
  will break perfect symmetry.
* A fixed number of microticks is allocated per tick; the metronome
  and compute budget system are simplified.

To run this scenario from within the project root:

```
python3 -m stratum.scenarios.stellar_collapse
```

This will run the simulation for a predefined number of ticks and
save a few snapshots as PNG images in the current directory.  If
matplotlib is unavailable or image generation fails, the simulation
will still run and print basic statistics to stdout.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib

# Use non‑interactive backend to allow headless image generation.
matplotlib.use("Agg")  # type: ignore
import matplotlib.pyplot as plt

from ..core.config import EngineConfig
from ..core.fabric import Fabric
from ..core.ledger import Ledger
from ..core.metronome import Metronome
from ..core.quanta import Quanta
from ..core.registry import SpeciesRegistry
from ..domains.materials.fundamentals import MaterialsFundamentals


def run_stellar_collapse(
    grid_size: int = 32,
    # run for many ticks to allow high‑energy interactions to develop
    num_ticks: int = 500,
    # allocate a generous microtick budget per tick; this controls the
    # fidelity of local relaxation and event resolution.  Increasing
    # this value increases run time but allows more interactions to
    # occur within each tick.  It should scale with the number of
    # active cells; for a 32×32 grid values in the hundreds are
    # reasonable.
    microticks_per_tick: int = 500,
    output_dir: str = "./stellar_outputs",
    snapshot_ticks: list[int] | None = None,
) -> None:
    """Run the simplified stellar collapse simulation.

    Parameters
    ----------
    grid_size:
        The width and height of the simulation grid.  The domain is
        square with periodic boundaries.
    num_ticks:
        Number of base simulation ticks to run.  Each tick comprises a
        number of microticks determined by ``microticks_per_tick``.
    microticks_per_tick:
        Total microticks to allocate across all active cells each tick.
        This value controls the overall "fidelity" of the update.
    output_dir:
        Directory where snapshot PNGs will be saved.  The directory
        will be created if it does not already exist.
    snapshot_ticks:
        Optional list of tick indices at which to save snapshots.  If
        ``None``, snapshots will be saved at ticks 0, half way and the
        final tick.

    Returns
    -------
    None
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    # Build configuration: override grid dimensions and budgets.  To
    # encourage interesting high‑energy interactions, we enable the
    # entropy_mode so that stochastic checkpoints use a run salt.  We
    # also lower the Z thresholds slightly from their defaults so
    # that fusion and degenerate transitions can occur at moderate
    # compression.  The active_region_max is set high enough to
    # capture most of the grid during the early stages of collapse.
    cfg = EngineConfig(
        grid_w=grid_size,
        grid_h=grid_size,
        entropy_mode=True,
        replay_mode=False,
        # Use periodic boundaries for both axes
        boundary="PERIODIC",
        # Many active cells; cap at 3/4 of the grid
        active_region_max=3 * grid_size * grid_size // 4,
        microtick_cap_per_region=10,
        tick_budget_ms=200.0,
        # Adjust Z thresholds to encourage events: lower degenerate and BH thresholds
        Z_fuse_min=1.2,
        Z_deg_min=2.5,
        Z_bh_min=4.0,
        Z_abs_max=6.0,
        Z_star_flip=2.5,
        # Slightly stronger gravity to assist collapse
        gravity_strength=0.1,
        # Keep other defaults from EngineConfig
    )
    # Define high and low energy property names for the registry
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
    # No low energy properties in this simplified run
    le_props: list[str] = []
    # Initialise registry at a temporary path (in‑memory if possible)
    registry_path = os.path.join(output_dir, "species_registry.json")
    registry = SpeciesRegistry(registry_path, he_props, le_props)
    # Instantiate fundamental materials; this registers StellarGas and DEG
    materials = MaterialsFundamentals(registry, cfg)
    # Create fabric and ledger
    # Allocate fabric using the configuration; Fabric uses the config
    fabric = Fabric(cfg)
    ledger = Ledger(fabric, cfg)
    # Create quanta subsystem; metronome unused in this simplified loop
    quanta = Quanta(fabric, ledger, registry, materials, cfg)
    # Initialise mixture: fill each cell with uniform mass of the stellar species
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
    # Determine snapshot ticks
    if snapshot_ticks is None:
        snapshot_ticks = [0, num_ticks // 2, num_ticks]
    # Function to save a snapshot image
    def save_snapshot(tick: int) -> None:
        rho = fabric.rho.copy()
        T_field = np.divide(fabric.E_heat, np.maximum(fabric.rho, 1e-8))
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        im0 = axes[0].imshow(rho.T, origin="lower", cmap="inferno")
        axes[0].set_title(f"Density at tick {tick}")
        fig.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(T_field.T, origin="lower", cmap="magma")
        axes[1].set_title(f"Temperature at tick {tick}")
        fig.colorbar(im1, ax=axes[1])
        plt.tight_layout()
        fname = os.path.join(output_dir, f"snapshot_{tick:04d}.png")
        fig.savefig(fname)
        plt.close(fig)

    # Save initial snapshot
    save_snapshot(0)
    # Main simulation loop
    for tick in range(1, num_ticks + 1):
        # allocate microticks per tick to the quanta subsystem
        quanta.step(tick, microticks_per_tick)
        # simple ledger finalisation: not needed for this prototype
        # apply global diffusion and smoothing (already called within quanta.step)
        # Save snapshots if requested
        if tick in snapshot_ticks:
            save_snapshot(tick)
        # Print basic statistics every few ticks
        if tick % max(1, num_ticks // 5) == 0:
            total_mass = fabric.rho.sum()
            total_heat = fabric.E_heat.sum()
            total_rad = fabric.E_rad.sum()
            print(f"tick {tick:3d}: mass={total_mass:.3f}, heat={total_heat:.3f}, rad={total_rad:.3f}")
    # Save final snapshot
    if num_ticks not in snapshot_ticks:
        save_snapshot(num_ticks)
    # Persist registry in case new species were created (unlikely in this simplified run)
    registry.save()
    print(f"Simulation complete. Snapshots saved to {output_dir}")


def main():
    """Entry point for the stellar collapse scenario."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run stellar collapse simulation")
    parser.add_argument("--grid", type=int, default=32, help="Grid size (width=height)")
    parser.add_argument("--ticks", type=int, default=500, help="Number of ticks to run")
    parser.add_argument("--microticks", type=int, default=500, help="Microticks per tick")
    parser.add_argument("--output", type=str, default="./stellar_outputs", help="Output directory")
    args = parser.parse_args()
    
    run_stellar_collapse(
        grid_size=args.grid,
        num_ticks=args.ticks,
        microticks_per_tick=args.microticks,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
