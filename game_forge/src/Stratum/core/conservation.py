"""
Conservation accounting system for Stratum.

This module provides detailed tracking of mass, energy, and momentum
conservation throughout the simulation. It implements the "accounting
surfaces" concept where sources, sinks, and boundary fluxes are
explicitly tracked to enable conservation verification.

Key Features:
- Per-tick source/sink ledger
- Boundary flux counters
- Conservation violation detection
- Diagnostic reporting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
from eidosian_core import eidosian

# Numerical stability constants for conservation calculations
# These limits are chosen to be well within float64 range (~1e308) while
# leaving headroom for intermediate calculations.

#: Minimum density threshold below which mass is considered negligible
RHO_EPSILON = 1e-12

#: Maximum density to prevent overflow in kinetic energy calculation
RHO_MAX_CLAMP = 1e150

#: Maximum momentum component before clamping (sqrt of max squared sum)
MOMENTUM_MAX_CLAMP = 1e75

#: Maximum kinetic energy per cell
KINETIC_ENERGY_MAX = 1e100


class FluxType(Enum):
    """Types of mass/energy flux that can occur."""
    BOUNDARY_OUTFLOW = "boundary_outflow"  # Mass leaving through open boundaries
    BOUNDARY_INFLOW = "boundary_inflow"    # Mass entering through boundaries
    BH_ABSORPTION = "bh_absorption"        # Mass absorbed by black holes
    SOURCE_INJECTION = "source_injection"  # External mass/energy injection
    NUMERICAL_LOSS = "numerical_loss"      # Mass lost to numerical clamping
    CONVERSION = "conversion"              # Energy form conversion (e.g., kinetic to heat)


@dataclass
class FluxRecord:
    """Record of a single flux event.
    
    Attributes:
        tick: Simulation tick when flux occurred.
        flux_type: Type of flux.
        cell: Grid cell (i, j) where flux occurred, or None for global fluxes.
        mass_delta: Change in mass (positive = gain, negative = loss).
        energy_delta: Change in energy.
        momentum_delta: Change in momentum (px, py).
        description: Human-readable description of the flux.
    """
    tick: int
    flux_type: FluxType
    cell: Optional[Tuple[int, int]]
    mass_delta: float
    energy_delta: float
    momentum_delta: Tuple[float, float]
    description: str = ""


@dataclass
class TickConservationStats:
    """Conservation statistics for a single tick.
    
    Attributes:
        tick: Simulation tick number.
        mass_initial: Total mass at start of tick.
        mass_final: Total mass at end of tick.
        energy_initial: Total energy at start of tick.
        energy_final: Total energy at end of tick.
        momentum_initial: Total momentum (px, py) at start of tick.
        momentum_final: Total momentum (px, py) at end of tick.
        boundary_mass_flux: Net mass flux through boundaries.
        boundary_energy_flux: Net energy flux through boundaries.
        bh_mass_absorbed: Mass absorbed by black holes.
        numerical_mass_loss: Mass lost to numerical clamping.
        conservation_error: Computed conservation error (should be near zero).
    """
    tick: int
    mass_initial: float = 0.0
    mass_final: float = 0.0
    energy_initial: float = 0.0
    energy_final: float = 0.0
    momentum_initial: Tuple[float, float] = (0.0, 0.0)
    momentum_final: Tuple[float, float] = (0.0, 0.0)
    boundary_mass_flux: float = 0.0
    boundary_energy_flux: float = 0.0
    bh_mass_absorbed: float = 0.0
    numerical_mass_loss: float = 0.0
    conservation_error: float = 0.0

    @eidosian()
    def compute_conservation_error(self) -> float:
        """Compute the mass conservation error.
        
        Conservation error = final - (initial - outflows + inflows - bh_absorbed - numerical_loss)
        Should be zero for perfect conservation.
        """
        expected_final = (
            self.mass_initial 
            - self.boundary_mass_flux 
            - self.bh_mass_absorbed 
            - self.numerical_mass_loss
        )
        self.conservation_error = self.mass_final - expected_final
        return self.conservation_error


class ConservationLedger:
    """Tracks conservation of mass, energy, and momentum.
    
    This class provides comprehensive accounting of all sources and sinks
    of conserved quantities in the simulation. It enables verification
    that the simulation conserves mass/energy within expected tolerances.
    
    Usage:
        ledger = ConservationLedger()
        ledger.begin_tick(tick, fabric)
        # ... simulation step ...
        ledger.record_boundary_flux(tick, (i, j), mass, energy)
        ledger.end_tick(tick, fabric)
        stats = ledger.get_tick_stats(tick)
    """

    def __init__(self, tolerance: float = 1e-6):
        """Initialize the conservation ledger.
        
        Args:
            tolerance: Threshold for conservation violation warnings.
        """
        self.tolerance = tolerance
        self.tick_stats: Dict[int, TickConservationStats] = {}
        self.flux_records: List[FluxRecord] = []
        self.boundary_flux_counters: Dict[str, float] = {
            'mass_out': 0.0,
            'mass_in': 0.0,
            'energy_out': 0.0,
            'energy_in': 0.0,
            'px_out': 0.0,
            'py_out': 0.0,
        }
        self.total_bh_absorbed: float = 0.0
        self.total_numerical_loss: float = 0.0
        self._current_tick_stats: Optional[TickConservationStats] = None

    @eidosian()
    def begin_tick(self, tick: int, fabric: 'Fabric') -> None:
        """Record initial state at the beginning of a tick.
        
        Args:
            tick: Current tick number.
            fabric: Fabric instance containing field data.
        """
        stats = TickConservationStats(tick=tick)
        stats.mass_initial = float(fabric.rho.sum() + fabric.BH_mass.sum())
        stats.energy_initial = float(
            fabric.E_heat.sum() + 
            fabric.E_rad.sum() + 
            self._compute_kinetic_energy(fabric)
        )
        stats.momentum_initial = (float(fabric.px.sum()), float(fabric.py.sum()))
        
        self._current_tick_stats = stats
        self.tick_stats[tick] = stats

    @eidosian()
    def end_tick(self, tick: int, fabric: 'Fabric') -> TickConservationStats:
        """Record final state at the end of a tick.
        
        Args:
            tick: Current tick number.
            fabric: Fabric instance containing field data.
            
        Returns:
            TickConservationStats for this tick.
        """
        stats = self._current_tick_stats
        if stats is None or stats.tick != tick:
            # Create new stats if begin_tick wasn't called
            stats = TickConservationStats(tick=tick)
            self.tick_stats[tick] = stats
        
        stats.mass_final = float(fabric.rho.sum() + fabric.BH_mass.sum())
        stats.energy_final = float(
            fabric.E_heat.sum() + 
            fabric.E_rad.sum() + 
            self._compute_kinetic_energy(fabric)
        )
        stats.momentum_final = (float(fabric.px.sum()), float(fabric.py.sum()))
        
        # Compute conservation error
        stats.compute_conservation_error()
        
        self._current_tick_stats = None
        return stats

    @eidosian()
    def record_boundary_flux(
        self,
        tick: int,
        cell: Tuple[int, int],
        mass_delta: float,
        energy_delta: float,
        momentum_delta: Tuple[float, float] = (0.0, 0.0),
        direction: str = "out"
    ) -> None:
        """Record mass/energy flux through a boundary cell.
        
        Args:
            tick: Current tick number.
            cell: Grid cell coordinates.
            mass_delta: Mass that crossed the boundary (positive = outflow).
            energy_delta: Energy that crossed the boundary.
            momentum_delta: Momentum that crossed the boundary.
            direction: "out" for outflow, "in" for inflow.
        """
        flux_type = FluxType.BOUNDARY_OUTFLOW if direction == "out" else FluxType.BOUNDARY_INFLOW
        sign = 1.0 if direction == "out" else -1.0
        
        record = FluxRecord(
            tick=tick,
            flux_type=flux_type,
            cell=cell,
            mass_delta=-sign * mass_delta,  # Negative for loss
            energy_delta=-sign * energy_delta,
            momentum_delta=(-sign * momentum_delta[0], -sign * momentum_delta[1]),
            description=f"Boundary {direction}flow at {cell}"
        )
        self.flux_records.append(record)
        
        # Update counters
        if direction == "out":
            self.boundary_flux_counters['mass_out'] += mass_delta
            self.boundary_flux_counters['energy_out'] += energy_delta
            self.boundary_flux_counters['px_out'] += momentum_delta[0]
            self.boundary_flux_counters['py_out'] += momentum_delta[1]
        else:
            self.boundary_flux_counters['mass_in'] += mass_delta
            self.boundary_flux_counters['energy_in'] += energy_delta
        
        # Update tick stats
        if self._current_tick_stats is not None:
            self._current_tick_stats.boundary_mass_flux += sign * mass_delta
            self._current_tick_stats.boundary_energy_flux += sign * energy_delta

    @eidosian()
    def record_bh_absorption(
        self,
        tick: int,
        cell: Tuple[int, int],
        mass: float,
        energy: float
    ) -> None:
        """Record mass/energy absorbed by a black hole.
        
        Args:
            tick: Current tick number.
            cell: Grid cell where BH is located.
            mass: Mass absorbed.
            energy: Energy absorbed.
        """
        record = FluxRecord(
            tick=tick,
            flux_type=FluxType.BH_ABSORPTION,
            cell=cell,
            mass_delta=-mass,
            energy_delta=-energy,
            momentum_delta=(0.0, 0.0),
            description=f"BH absorption at {cell}"
        )
        self.flux_records.append(record)
        self.total_bh_absorbed += mass
        
        if self._current_tick_stats is not None:
            self._current_tick_stats.bh_mass_absorbed += mass

    @eidosian()
    def record_numerical_loss(
        self,
        tick: int,
        cell: Optional[Tuple[int, int]],
        mass_lost: float,
        reason: str = "clamping"
    ) -> None:
        """Record mass lost due to numerical operations (e.g., negative clamping).
        
        Args:
            tick: Current tick number.
            cell: Grid cell where loss occurred, or None for global.
            mass_lost: Amount of mass lost.
            reason: Reason for the loss.
        """
        record = FluxRecord(
            tick=tick,
            flux_type=FluxType.NUMERICAL_LOSS,
            cell=cell,
            mass_delta=-mass_lost,
            energy_delta=0.0,
            momentum_delta=(0.0, 0.0),
            description=f"Numerical loss ({reason}) at {cell}"
        )
        self.flux_records.append(record)
        self.total_numerical_loss += mass_lost
        
        if self._current_tick_stats is not None:
            self._current_tick_stats.numerical_mass_loss += mass_lost

    @eidosian()
    def record_source_injection(
        self,
        tick: int,
        cell: Tuple[int, int],
        mass: float,
        energy: float = 0.0
    ) -> None:
        """Record external mass/energy injection.
        
        Args:
            tick: Current tick number.
            cell: Grid cell where injection occurred.
            mass: Mass injected.
            energy: Energy injected.
        """
        record = FluxRecord(
            tick=tick,
            flux_type=FluxType.SOURCE_INJECTION,
            cell=cell,
            mass_delta=mass,
            energy_delta=energy,
            momentum_delta=(0.0, 0.0),
            description=f"Source injection at {cell}"
        )
        self.flux_records.append(record)

    @eidosian()
    def get_tick_stats(self, tick: int) -> Optional[TickConservationStats]:
        """Get conservation statistics for a specific tick.
        
        Args:
            tick: Tick number.
            
        Returns:
            TickConservationStats or None if tick not recorded.
        """
        return self.tick_stats.get(tick)

    @eidosian()
    def get_conservation_summary(self) -> Dict[str, float]:
        """Get summary of conservation accounting.
        
        Returns:
            Dictionary with conservation summary statistics.
        """
        return {
            'total_boundary_mass_out': self.boundary_flux_counters['mass_out'],
            'total_boundary_mass_in': self.boundary_flux_counters['mass_in'],
            'net_boundary_mass_flux': (
                self.boundary_flux_counters['mass_out'] - 
                self.boundary_flux_counters['mass_in']
            ),
            'total_bh_absorbed': self.total_bh_absorbed,
            'total_numerical_loss': self.total_numerical_loss,
            'max_conservation_error': max(
                (abs(s.conservation_error) for s in self.tick_stats.values()),
                default=0.0
            ),
        }

    @eidosian()
    def check_conservation(self, tolerance: Optional[float] = None) -> Tuple[bool, str]:
        """Check if conservation is maintained within tolerance.
        
        Args:
            tolerance: Conservation tolerance. Uses instance default if None.
            
        Returns:
            Tuple of (is_conserved, message).
        """
        tol = tolerance if tolerance is not None else self.tolerance
        
        violations = []
        for tick, stats in self.tick_stats.items():
            if abs(stats.conservation_error) > tol:
                violations.append(f"Tick {tick}: error={stats.conservation_error:.2e}")
        
        if violations:
            return False, f"Conservation violations: {', '.join(violations[:5])}"
        
        return True, "Conservation maintained within tolerance"

    def _compute_kinetic_energy(self, fabric: 'Fabric') -> float:
        """Compute total kinetic energy in the fabric.
        
        Args:
            fabric: Fabric instance.
            
        Returns:
            Total kinetic energy.
        """
        # KE = p^2 / (2*rho), but only where rho > 0
        rho = fabric.rho
        px = fabric.px
        py = fabric.py
        
        # Avoid division by zero and overflow
        mask = (rho > RHO_EPSILON) & np.isfinite(rho)
        ke = np.zeros_like(rho)
        
        # Clamp momentum to prevent overflow in squaring
        px_safe = np.clip(px[mask], -MOMENTUM_MAX_CLAMP, MOMENTUM_MAX_CLAMP)
        py_safe = np.clip(py[mask], -MOMENTUM_MAX_CLAMP, MOMENTUM_MAX_CLAMP)
        rho_safe = np.clip(rho[mask], RHO_EPSILON, RHO_MAX_CLAMP)
        
        ke[mask] = (px_safe**2 + py_safe**2) / (2.0 * rho_safe)
        
        # Clamp any infinities or NaNs
        ke = np.where(np.isfinite(ke), ke, 0.0)
        ke = np.clip(ke, 0, KINETIC_ENERGY_MAX)
        
        return float(np.sum(ke))

    @eidosian()
    def clear(self) -> None:
        """Clear all recorded data."""
        self.tick_stats.clear()
        self.flux_records.clear()
        self.boundary_flux_counters = {
            'mass_out': 0.0,
            'mass_in': 0.0,
            'energy_out': 0.0,
            'energy_in': 0.0,
            'px_out': 0.0,
            'py_out': 0.0,
        }
        self.total_bh_absorbed = 0.0
        self.total_numerical_loss = 0.0
        self._current_tick_stats = None
