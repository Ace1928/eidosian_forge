"""
Real‑time screensaver for the Stratum simulation (density + temperature side‑by‑side).

This module implements a live visualiser for the Stratum engine using Pygame.  It
runs a stellar collapse simulation indefinitely and displays both the mass
density and the temperature fields in a single window.  It also adapts the
simulation’s level of detail (LOD) at runtime based on the achieved frame
rate.  This allows larger grids to be displayed at fluid frame rates by
reducing the number of active regions and micro‑tick iterations.

Features:

* Auto‑sized grids: If ``--grid`` is not provided or set to 0, the grid size
  is chosen based on the display resolution such that one simulation cell
  maps to one pixel per panel.  Two panels are shown: density on the left
  and temperature on the right.
* Density and temperature visualisation: The left panel shows the mass
  density; the right shows the temperature proxy (heat per unit mass).  Both
  fields are normalised independently to 0..255 and optionally log‑scaled
  to improve contrast.
* Dynamic LOD: A simple EMA‑based controller adjusts ``active_region_max``
  and ``microtick_cap_per_region`` during the run to maintain a target
  frame rate.  You can bias this controller toward quality or speed with
  runtime controls (keys 1/2).  Target FPS can be increased/decreased with
  +/- keys.
* Controls:
    - ``ESC`` or ``Q``: quit
    - ``SPACE``: pause/resume simulation
    - ``R``: reseed and restart the simulation
    - ``TAB``: toggle log scaling for visualisation
    - ``1``/``2``: bias dynamic LOD toward speed/quality
    - ``+``/``-``: raise or lower the target FPS

Usage examples::

    python3 -m stratum.scenarios.stellar_screensaver
    python3 -m stratum.scenarios.stellar_screensaver --grid 256 --fps 60
    python3 -m stratum.scenarios.stellar_screensaver --grid 0  # auto grid

Dependencies: This script requires Pygame (``pip install pygame``) in
addition to NumPy.  It assumes that the Stratum engine components are
available in the ``stratum.core`` and ``stratum.domains.materials``
namespaces.

"""

from __future__ import annotations

import argparse
import os
import time
import tempfile
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pygame

from ..core.config import EngineConfig
from ..core.fabric import Fabric
from ..core.ledger import Ledger
from ..core.quanta import Quanta
from ..core.registry import SpeciesRegistry
from ..domains.materials.fundamentals import MaterialsFundamentals


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def _clamp(x: float, lo: float, hi: float) -> float:
    """Return x clamped to the inclusive range [lo, hi]."""
    return lo if x < lo else hi if x > hi else x


def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Safely divide arrays elementwise, avoiding division by zero."""
    return np.divide(a, np.maximum(b, eps), out=np.zeros_like(a), where=(b > 0))


def _normalize_to_u8(field: np.ndarray, log_scale: bool = True) -> np.ndarray:
    """
    Normalize a 2‑D scalar field to the range [0, 255] as ``uint8``.

    If ``log_scale`` is True, apply ``log1p`` to compress dynamic range
    before normalising.  Using float32 here can overflow if values are
    enormous; we convert to float64 for safety and clamp negatives to zero.
    """
    f = field.astype(np.float64, copy=False)
    if log_scale:
        f = np.log1p(np.maximum(f, 0.0))
    maxv = float(np.max(f))
    if not np.isfinite(maxv) or maxv <= 0.0:
        return np.zeros_like(f, dtype=np.uint8)
    scaled = f / maxv
    return (scaled * 255.0).clip(0.0, 255.0).astype(np.uint8)


def _make_rgb(gray_u8: np.ndarray) -> np.ndarray:
    """Stack a greyscale array into an RGB image."""
    return np.repeat(gray_u8[..., np.newaxis], 3, axis=-1)


def _display_size() -> Tuple[int, int]:
    """Return the current display resolution in pixels."""
    info = pygame.display.Info()
    return int(info.current_w), int(info.current_h)


# -----------------------------------------------------------------------------
# Dynamic LOD controller
# -----------------------------------------------------------------------------


@dataclass
class DynamicLOD:
    """
    Simple exponential moving‑average controller for adjusting level‑of‑detail.

    Attributes:
    - target_fps: desired frames per second.
    - lod_quality_bias: in [0,1]; 0 biases toward speed, 1 toward quality.
    - min_active_regions / max_active_regions: bounds for active regions.
    - min_micro_cap / max_micro_cap: bounds for microtick caps.
    """

    target_fps: float
    lod_quality_bias: float = 0.5
    min_active_regions: int = 64
    max_active_regions: int = 1_000_000
    min_micro_cap: int = 1
    max_micro_cap: int = 64

    ema_fps: float = 0.0
    ema_alpha: float = 0.08

    def update(self, measured_fps: float) -> None:
        """Update the EMA of the measured FPS."""
        if self.ema_fps <= 0.0:
            self.ema_fps = measured_fps
        else:
            self.ema_fps = (1.0 - self.ema_alpha) * self.ema_fps + self.ema_alpha * measured_fps

    def apply(self, cfg: EngineConfig, base_active: int, base_micro_cap: int) -> None:
        """
        Adjust ``cfg.active_region_max`` and ``cfg.microtick_cap_per_region``
        based on the current FPS.  The adjustment factor is computed from the
        ratio of measured FPS to target FPS.  When FPS is below target,
        settings are scaled down; when above, they are scaled up.  The
        ``lod_quality_bias`` modulates the aggressiveness of these changes.
        """
        fps = max(self.ema_fps, 1e-6)
        ratio = fps / max(self.target_fps, 1e-6)

        down_aggr = 0.25 + (1.0 - self.lod_quality_bias) * 0.75
        up_aggr = 0.10 + self.lod_quality_bias * 0.20
        if ratio < 1.0:
            scale = ratio ** down_aggr
        else:
            scale = ratio ** up_aggr

        new_active = int(_clamp(base_active * scale, self.min_active_regions, self.max_active_regions))
        new_micro = int(_clamp(base_micro_cap * scale, self.min_micro_cap, self.max_micro_cap))

        # Avoid jitter: update only when significantly different.
        if abs(new_active - getattr(cfg, "active_region_max")) > max(8, int(0.03 * new_active)):
            cfg.active_region_max = new_active
        if abs(new_micro - getattr(cfg, "microtick_cap_per_region")) >= 1:
            cfg.microtick_cap_per_region = new_micro


# -----------------------------------------------------------------------------
# Engine construction helper
# -----------------------------------------------------------------------------


def _build_engine(
    grid_w: int,
    grid_h: int,
    microticks_per_tick: int,
    lod_factor: float,
    gravity_strength: float,
    seed: int,
) -> Tuple[EngineConfig, Fabric, Ledger, SpeciesRegistry, MaterialsFundamentals, Quanta, int, int]:
    """Initialise a Stratum engine instance with the specified parameters."""

    # Mixture top K from LOD factor; ensure at least 1.
    base_top_k = 8
    mixture_top_k = max(1, int(base_top_k * lod_factor))

    # Base active region count and microtick cap for DynamicLOD reference.
    base_active_regions = max(64, int(0.75 * grid_w * grid_h * lod_factor))
    base_micro_cap = max(1, int(12 * lod_factor))

    # Build configuration.  Provide defaults for any extra fields in EngineConfig.
    cfg = EngineConfig(
        grid_w=grid_w,
        grid_h=grid_h,
        base_seed=seed if hasattr(EngineConfig, "base_seed") else 0,
        entropy_mode=True,
        replay_mode=False,
        boundary="PERIODIC",
        active_region_max=base_active_regions,
        microtick_cap_per_region=base_micro_cap,
        tick_budget_ms=200.0,

        # Regime thresholds tuned for dynamic screensaver; tweak as desired.
        Z_fuse_min=1.2,
        Z_deg_min=2.5,
        Z_bh_min=4.8,
        Z_abs_max=6.5,
        Z_star_flip=2.5,

        gravity_strength=gravity_strength,
        mixture_top_k=mixture_top_k,
    )

    # Create an on-disk registry path in a temporary directory.  SpeciesRegistry
    # expects a valid file path for persistence.  We allocate a temp file and
    # do not clean it up automatically to avoid deleting after the run; this
    # has negligible footprint and allows introspection if desired.
    tmpdir = tempfile.mkdtemp(prefix="stratum_registry_")
    reg_path = os.path.join(tmpdir, "registry.json")

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

    registry = SpeciesRegistry(reg_path, he_props, [])
    materials = MaterialsFundamentals(registry, cfg)
    fabric = Fabric(cfg)
    ledger = Ledger(fabric, cfg)
    quanta = Quanta(fabric, ledger, registry, materials, cfg)

    # Initialise the grid with near-uniform density, no momentum, and high heat.
    stellar_id = materials.stellar_species.id
    rng = np.random.default_rng(seed)
    fabric.rho[:, :] = 1.0 + 0.02 * rng.standard_normal(size=(grid_w, grid_h)).astype(np.float32)
    fabric.rho[:, :] = np.maximum(fabric.rho, 0.05)
    fabric.px[:, :] = 0.0
    fabric.py[:, :] = 0.0
    fabric.E_heat[:, :] = 8.0
    fabric.E_rad[:, :] = 0.0

    # Fill species mixture arrays
    for i in range(grid_w):
        row = fabric.mixtures[i]
        for j in range(grid_h):
            mix = row[j]
            mix.species_ids = [stellar_id]
            mix.masses = [float(fabric.rho[i, j])]

    return cfg, fabric, ledger, registry, materials, quanta, base_active_regions, base_micro_cap


# -----------------------------------------------------------------------------
# Main screensaver loop
# -----------------------------------------------------------------------------


def run_stellar_screensaver(
    grid_size: int = 0,
    microticks_per_tick: int = 50,
    scale: int = 1,
    lod_factor: float = 1.0,
    fps: int = 30,
    gravity_strength: float = 0.05,
) -> None:
    """
    Execute the Stratum simulation in a real‑time Pygame visualiser.

    Parameters
    ----------
    grid_size : int
        If >0, sets the simulation grid to this size (both width and height).
        If <=0, auto‑sizes the grid to half the display width (per panel).
    microticks_per_tick : int
        Number of micro‑ticks per simulation tick.  Higher values yield
        more accurate event resolution but reduce throughput.
    scale : int
        The pixel size of each simulation cell when rendering.  A scale of 1
        maps one cell to one pixel per panel.  Larger values produce
        chunkier pixels and bigger windows.
    lod_factor : float
        Initial LOD scaling factor in (0,1].  Smaller values track fewer
        species and use fewer active regions and micro‑ticks per region.
    fps : int
        Target frames per second.  Dynamic LOD attempts to maintain this.
    gravity_strength : float
        Strength of the gravitational attraction.  Adjust to control the
        collapse rate.  Higher values lead to faster clumping.
    """

    pygame.init()
    pygame.font.init()
    pygame.display.set_caption("Stratum — Living Screensaver (Density | Temperature)")
    font = pygame.font.Font(None, 22)

    # Determine grid dimensions and window size
    display_w, display_h = _display_size()
    if grid_size and grid_size > 0:
        grid_w = int(grid_size)
        grid_h = int(grid_size)
        win_w = 2 * grid_w * max(1, scale)
        win_h = grid_h * max(1, scale)
        flags = 0
    else:
        # Auto grid: fit panels horizontally across the display.  Each panel
        # occupies half the screen width by default with scale=1.  Scale can
        # enlarge cells if provided.
        grid_w = max(64, display_w // (2 * max(scale, 1)))
        grid_h = max(64, display_h // max(scale, 1))
        win_w = 2 * grid_w * max(1, scale)
        win_h = grid_h * max(1, scale)
        flags = pygame.NOFRAME if (win_w <= display_w and win_h <= display_h) else pygame.RESIZABLE

    screen = pygame.display.set_mode((win_w, win_h), flags)
    clock = pygame.time.Clock()

    # Build engine and dynamic LOD controller
    seed = int(time.time()) & 0xFFFFFFFF
    cfg, fabric, ledger, registry, materials, quanta, base_active, base_micro_cap = _build_engine(
        grid_w,
        grid_h,
        microticks_per_tick,
        _clamp(lod_factor, 0.05, 1.0),
        gravity_strength,
        seed,
    )
    dyn = DynamicLOD(target_fps=float(fps), lod_quality_bias=0.55,
                     min_active_regions=max(64, (grid_w * grid_h) // 200),
                     max_active_regions=grid_w * grid_h,
                     min_micro_cap=1,
                     max_micro_cap=max(8, base_micro_cap * 4))

    # Visualisation settings
    log_scale = True
    paused = False
    tick = 0
    last_fps_update = time.time()
    frames_since = 0

    def restart() -> None:
        nonlocal cfg, fabric, ledger, registry, materials, quanta, base_active, base_micro_cap, seed, tick
        seed = int(time.time()) & 0xFFFFFFFF
        cfg, fabric, ledger, registry, materials, quanta, base_active, base_micro_cap = _build_engine(
            grid_w,
            grid_h,
            microticks_per_tick,
            _clamp(lod_factor, 0.05, 1.0),
            gravity_strength,
            seed,
        )
        tick = 0

    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                    break
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r:
                    restart()
                if event.key == pygame.K_TAB:
                    log_scale = not log_scale
                if event.key == pygame.K_1:
                    dyn.lod_quality_bias = _clamp(dyn.lod_quality_bias - 0.10, 0.0, 1.0)
                if event.key == pygame.K_2:
                    dyn.lod_quality_bias = _clamp(dyn.lod_quality_bias + 0.10, 0.0, 1.0)
                if event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    dyn.target_fps = _clamp(dyn.target_fps + 5.0, 5.0, 240.0)
                if event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS):
                    dyn.target_fps = _clamp(dyn.target_fps - 5.0, 5.0, 240.0)

        if not running:
            break

        # Advance simulation
        if not paused:
            tick += 1
            dyn.apply(cfg, base_active, base_micro_cap)
            quanta.step(tick, microticks_per_tick)

        # Extract fields
        rho = fabric.rho
        temp = _safe_div(fabric.E_heat, rho)

        # Convert to images
        rho_u8 = _normalize_to_u8(rho, log_scale)
        temp_u8 = _normalize_to_u8(temp, log_scale)
        rho_rgb = _make_rgb(rho_u8)
        temp_rgb = _make_rgb(temp_u8)
        rho_surf = pygame.surfarray.make_surface(rho_rgb.transpose(1, 0, 2))
        temp_surf = pygame.surfarray.make_surface(temp_rgb.transpose(1, 0, 2))
        if scale != 1:
            rho_surf = pygame.transform.scale(rho_surf, (grid_w * scale, grid_h * scale))
            temp_surf = pygame.transform.scale(temp_surf, (grid_w * scale, grid_h * scale))

        # Render
        screen.fill((0, 0, 0))
        screen.blit(rho_surf, (0, 0))
        screen.blit(temp_surf, (grid_w * scale, 0))

        # HUD
        try:
            fps_now = clock.get_fps()
            hud_text = (
                f"tick={tick} fps={fps_now:5.1f} target={dyn.target_fps:4.0f} "
                f"active={cfg.active_region_max} microcap={cfg.microtick_cap_per_region} "
                f"mixK={getattr(cfg, 'mixture_top_k', '?')} log={'on' if log_scale else 'off'} "
                f"bias={dyn.lod_quality_bias:0.2f}"
            )
            hud = font.render(hud_text, True, (220, 220, 220))
            screen.blit(hud, (10, 10))
            lbl_density = font.render("Density", True, (240, 240, 240))
            lbl_temp = font.render("Temperature", True, (240, 240, 240))
            screen.blit(lbl_density, (10, 34))
            screen.blit(lbl_temp, (grid_w * scale + 10, 34))
        except Exception:
            pass

        pygame.display.flip()
        clock.tick(int(dyn.target_fps))

        # Update measured FPS for dynamic LOD
        frames_since += 1
        now = time.time()
        if now - last_fps_update >= 0.5:
            measured = frames_since / max(now - last_fps_update, 1e-6)
            dyn.update(measured)
            frames_since = 0
            last_fps_update = now

    pygame.quit()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stratum real‑time screensaver (density and temperature panels)"
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=0,
        help="Grid size (width=height). Use 0 to auto‑fit to display.",
    )
    parser.add_argument(
        "--microticks",
        type=int,
        default=50,
        help="Microticks per simulation tick (inner relaxation steps).",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help="Pixel scale factor per cell (default 1 = one cell per pixel).",
    )
    parser.add_argument(
        "--lod",
        type=float,
        default=1.0,
        help="Initial level of detail factor (0.05 <= LOD <= 1.0).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target frames per second for dynamic LOD.",
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=0.05,
        help="Gravitational strength (higher = faster collapse).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the stellar screensaver."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    run_stellar_screensaver(
        grid_size=args.grid,
        microticks_per_tick=max(1, args.microticks),
        scale=max(1, args.scale),
        lod_factor=_clamp(args.lod, 0.05, 1.0),
        fps=max(5, args.fps),
        gravity_strength=float(args.gravity),
    )


if __name__ == "__main__":
    main()
