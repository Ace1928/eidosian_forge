"""
Simulation configuration definitions.

This module defines configuration dataclasses used to parameterise the
simulation. Configurations are defined with explicit defaults so that
test runs can be created easily without requiring the user to supply
values for every field. See ``EngineConfig`` for the top‑level
configuration consumed by the Stratum engine.

Key Configuration Areas:
- Grid and boundary settings
- Determinism and reproducibility modes
- Time model and stability constraints
- Physics coefficients
- Performance tuning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from eidosian_core import eidosian


class DeterminismMode(Enum):
    """Determinism guarantees provided by the simulation.
    
    STRICT_DETERMINISTIC: Same seed + same version + same config => bitwise-identical fields.
        - Fixed patch ordering (lexicographic)
        - Deterministic RNG stream partitioning
        - No multithreading
        - Controlled floating-point rounding
        
    REPLAY_DETERMINISTIC: Same seed yields statistically similar outcomes.
        - Allows adaptive scheduling
        - Results may vary slightly across platforms
        
    REALTIME_ADAPTIVE: Prioritizes performance over reproducibility.
        - LOD changes allowed
        - Budget-driven scheduling
        - Parallel execution permitted
    """
    STRICT_DETERMINISTIC = "strict"
    REPLAY_DETERMINISTIC = "replay"
    REALTIME_ADAPTIVE = "adaptive"


class NegativeDensityPolicy(Enum):
    """Policy for handling negative density that may arise from numerical errors.
    
    CLAMP_ZERO: Clamp to zero and lose the mass (not conserving).
    CLAMP_AND_COMPENSATE: Clamp to zero and add deficit to neighboring cells.
    FAIL_FAST: Raise an error immediately (for debugging).
    """
    CLAMP_ZERO = "clamp_zero"
    CLAMP_AND_COMPENSATE = "clamp_compensate"
    FAIL_FAST = "fail"


@dataclass
class EngineConfig:
    """Top level configuration for Stratum engine runs.

    Fields correspond closely to the configuration table described in
    the specification. Where appropriate, fields include defaults that
    work reasonably for small demonstrations. Users are encouraged to
    override these as needed when constructing a scenario.
    
    Attributes:
        grid_w, grid_h: Grid dimensions (cells).
        dx: Grid spacing (physical units). Defaults to 1.0.
        dt_tick: Time per tick (physical units). Defaults to 1.0.
        
        determinism_mode: Level of determinism guarantee.
        base_seed: Base seed for reproducible random generation.
        entropy_mode: If True, inject run-specific salt for fuzziness.
        replay_mode: If True, record entropy draws for exact replay.
        
        boundary: Boundary condition ("PERIODIC", "REFLECTIVE", or "OPEN").
        negative_density_policy: How to handle negative density.
        
        cfl_safety_factor: Safety factor for CFL-like stability (0 < factor <= 1).
        diffusion_stability_limit: Maximum D*dt/dx^2 before degradation.
        advection_stability_limit: Maximum |v|*dt/dx before degradation.
    """

    # Grid dimensions and physical spacing
    grid_w: int = 128
    grid_h: int = 128
    dx: float = 1.0  # Grid spacing in physical units
    dt_tick: float = 1.0  # Time per tick in physical units

    # Determinism and reproducibility
    determinism_mode: DeterminismMode = DeterminismMode.REPLAY_DETERMINISTIC
    base_seed: int = 42
    entropy_mode: bool = False  # if True, inject run salt for fuzziness
    replay_mode: bool = False   # if True, record entropy draws for exact replay

    # Boundary condition
    boundary: str = "PERIODIC"  # PERIODIC, REFLECTIVE or OPEN
    negative_density_policy: NegativeDensityPolicy = NegativeDensityPolicy.CLAMP_ZERO

    # Propagation speeds (cells per tick)
    v_max: float = 5.0
    v_influence: float = 2.0
    v_radiation: float = 5.0

    # Time and compute budgets
    tick_budget_ms: float = 50.0  # approximate time slice per tick
    degrade_first: bool = True
    microtick_cap_per_region: int = 10
    active_region_max: int = 2048

    # Stability constraints (CFL-like bounds)
    cfl_safety_factor: float = 0.5  # Safety factor for stability (0 < x <= 1)
    diffusion_stability_limit: float = 0.25  # Max D*dt/dx^2 for 2D explicit diffusion
    advection_stability_limit: float = 0.5  # Max |v|*dt/dx for advection

    # Mixture handling
    mixture_top_k: int = 4
    mixture_eps_merge: float = 1e-6

    # Physics coefficients
    gravity_strength: float = 0.05
    eos_gamma: float = 2.0
    thermal_pressure_coeff: float = 0.1
    repulsion_k: float = 50.0
    repulsion_n: float = 2.0
    shock_k: float = 0.2
    viscosity_global: float = 0.005

    # Radiation and absorption
    rad_to_heat_absorb_rate: float = 0.01

    # Regime thresholds for Z index
    Z_fuse_min: float = 1.5
    Z_deg_min: float = 3.0
    Z_bh_min: float = 4.5
    Z_abs_max: float = 6.0
    Z_star_flip: float = 2.5

    # Chemistry gating thresholds and tick ratio
    chemistry_tick_ratio: int = 5
    Z_chem_max: float = 1.0
    T_chem_max: float = 0.5

    # Black hole parameters
    EH_k: float = 0.5
    BH_absorb_energy_scale: float = 0.1

    # Stability coefficients for high‑energy stability function
    stability_low_coeff: float = 1.0
    stability_high_coeff: float = 1.0
    stability_temp_coeff: float = 0.5

    # Conservation enforcement
    enforce_mass_conservation: bool = False  # If True, renormalize mass after each tick
    enforce_energy_bounds: bool = True  # If True, clamp negative energies

    # World law version for replay compatibility
    world_law_version: str = "1.0.0"

    # Derived parameters reserved for future use
    extras: dict = field(default_factory=dict)

    @eidosian()
    def to_dict(self) -> dict:
        """Return a dict representation of the configuration.

        Useful for serialisation or interfacing with dynamic
        configuration loaders.
        """
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Enum):
                result[k] = v.value
            else:
                result[k] = v
        return result

    @eidosian()
    def compute_stability_dt(self, max_diffusivity: float = 0.1, max_velocity: float = 1.0) -> float:
        """Compute the maximum stable timestep based on CFL-like constraints.
        
        Args:
            max_diffusivity: Maximum diffusion coefficient in the simulation.
            max_velocity: Maximum velocity magnitude in the simulation.
            
        Returns:
            Maximum stable dt value.
        """
        # Diffusion constraint: D*dt/dx^2 <= limit
        dt_diffusion = float('inf')
        if max_diffusivity > 0:
            dt_diffusion = self.diffusion_stability_limit * self.dx * self.dx / max_diffusivity
        
        # Advection constraint: |v|*dt/dx <= limit
        dt_advection = float('inf')
        if max_velocity > 0:
            dt_advection = self.advection_stability_limit * self.dx / max_velocity
        
        # Take minimum with safety factor
        return self.cfl_safety_factor * min(dt_diffusion, dt_advection)

    @eidosian()
    def is_deterministic(self) -> bool:
        """Check if the configuration enforces strict determinism."""
        return self.determinism_mode == DeterminismMode.STRICT_DETERMINISTIC