"""
Diagnostics and invariant checking for Stratum.

This module provides tools for detecting numerical issues, enforcing
invariants, and generating diagnostic reports. It implements the
"fail-fast" philosophy where numerical instabilities are detected
early and reported with context for debugging.

Key Features:
- NaN/Inf detection with location tracking
- Invariant enforcement (non-negative density, etc.)
- Diagnostic snapshots for debugging
- Field statistics and histograms
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import warnings

import numpy as np
from eidosian_core import eidosian


class InvariantViolation(Exception):
    """Exception raised when a simulation invariant is violated.
    
    This exception provides detailed context about the violation including
    the field, location, and values involved.
    """
    
    def __init__(
        self,
        message: str,
        field_name: str,
        tick: int,
        locations: Optional[List[Tuple[int, int]]] = None,
        values: Optional[List[float]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.field_name = field_name
        self.tick = tick
        self.locations = locations or []
        self.values = values or []
        self.context = context or {}
        
        detail = f"{message} in field '{field_name}' at tick {tick}"
        if locations:
            detail += f"\n  Locations (first 5): {locations[:5]}"
        if values:
            detail += f"\n  Values (first 5): {values[:5]}"
        
        super().__init__(detail)


class DiagnosticLevel(Enum):
    """Level of diagnostic checking to perform."""
    NONE = "none"           # No checks (maximum performance)
    BASIC = "basic"         # Check for NaN/Inf only
    STANDARD = "standard"   # Check invariants (non-negative, etc.)
    STRICT = "strict"       # All checks including bounds


@dataclass
class FieldStats:
    """Statistics for a single field.
    
    Attributes:
        name: Field name.
        min_val: Minimum value in field.
        max_val: Maximum value in field.
        mean_val: Mean value.
        std_val: Standard deviation.
        nan_count: Number of NaN values.
        inf_count: Number of Inf values.
        negative_count: Number of negative values (for non-negative fields).
    """
    name: str
    min_val: float
    max_val: float
    mean_val: float
    std_val: float
    nan_count: int
    inf_count: int
    negative_count: int = 0


@dataclass
class DiagnosticSnapshot:
    """Snapshot of simulation state for debugging.
    
    Attributes:
        tick: Tick when snapshot was taken.
        field_stats: Statistics for each field.
        total_mass: Total mass in simulation.
        total_energy: Total energy in simulation.
        issues: List of detected issues.
    """
    tick: int
    field_stats: Dict[str, FieldStats]
    total_mass: float
    total_energy: float
    issues: List[str] = field(default_factory=list)


class Diagnostics:
    """Diagnostic checker for simulation fields.
    
    This class provides methods to check fields for numerical issues
    and enforce invariants. It can operate in different levels of
    strictness depending on performance requirements.
    
    Example:
        diag = Diagnostics(level=DiagnosticLevel.STANDARD)
        diag.check_field(fabric.rho, "rho", tick, non_negative=True)
        diag.check_all_fields(fabric, tick)
        snapshot = diag.take_snapshot(fabric, tick)
    """
    
    def __init__(
        self,
        level: DiagnosticLevel = DiagnosticLevel.STANDARD,
        fail_fast: bool = True,
        max_issues: int = 100,
        warn_on_issue: bool = False
    ):
        """Initialize diagnostics.
        
        Args:
            level: Level of checking to perform.
            fail_fast: If True, raise exception on first issue.
            max_issues: Maximum issues to track before stopping.
            warn_on_issue: If True, emit warnings for non-fatal issues.
        """
        self.level = level
        self.fail_fast = fail_fast
        self.max_issues = max_issues
        self.warn_on_issue = warn_on_issue
        self.issues: List[str] = []
        self.last_stable_tick: int = 0

    @eidosian()
    def check_field(
        self,
        field: np.ndarray,
        name: str,
        tick: int,
        non_negative: bool = False,
        max_value: Optional[float] = None,
        min_value: Optional[float] = None
    ) -> bool:
        """Check a field for issues.
        
        Args:
            field: Field array to check.
            name: Field name for error messages.
            tick: Current tick.
            non_negative: If True, check that all values >= 0.
            max_value: Optional maximum allowed value.
            min_value: Optional minimum allowed value.
            
        Returns:
            True if field passes all checks.
            
        Raises:
            InvariantViolation: If fail_fast is True and an issue is found.
        """
        if self.level == DiagnosticLevel.NONE:
            return True
        
        is_ok = True
        
        # Check for NaN
        nan_mask = np.isnan(field)
        if np.any(nan_mask):
            nan_locs = list(zip(*np.where(nan_mask)))
            is_ok = False
            self._report_issue(
                f"NaN detected in {name}",
                name, tick, nan_locs[:10], None
            )
        
        # Check for Inf
        inf_mask = np.isinf(field)
        if np.any(inf_mask):
            inf_locs = list(zip(*np.where(inf_mask)))
            inf_vals = [float(field[loc]) for loc in inf_locs[:10]]
            is_ok = False
            self._report_issue(
                f"Inf detected in {name}",
                name, tick, inf_locs[:10], inf_vals
            )
        
        if self.level in (DiagnosticLevel.STANDARD, DiagnosticLevel.STRICT):
            # Check non-negative constraint
            if non_negative:
                neg_mask = field < 0
                if np.any(neg_mask):
                    neg_locs = list(zip(*np.where(neg_mask)))
                    neg_vals = [float(field[loc]) for loc in neg_locs[:10]]
                    is_ok = False
                    self._report_issue(
                        f"Negative values in non-negative field {name}",
                        name, tick, neg_locs[:10], neg_vals
                    )
        
        if self.level == DiagnosticLevel.STRICT:
            # Check bounds
            if max_value is not None:
                over_mask = field > max_value
                if np.any(over_mask):
                    over_locs = list(zip(*np.where(over_mask)))
                    over_vals = [float(field[loc]) for loc in over_locs[:10]]
                    is_ok = False
                    self._report_issue(
                        f"Values exceed max ({max_value}) in {name}",
                        name, tick, over_locs[:10], over_vals
                    )
            
            if min_value is not None:
                under_mask = field < min_value
                if np.any(under_mask):
                    under_locs = list(zip(*np.where(under_mask)))
                    under_vals = [float(field[loc]) for loc in under_locs[:10]]
                    is_ok = False
                    self._report_issue(
                        f"Values below min ({min_value}) in {name}",
                        name, tick, under_locs[:10], under_vals
                    )
        
        if is_ok:
            self.last_stable_tick = tick
        
        return is_ok

    @eidosian()
    def check_all_fields(self, fabric: 'Fabric', tick: int) -> bool:
        """Check all fabric fields for issues.
        
        Args:
            fabric: Fabric instance to check.
            tick: Current tick.
            
        Returns:
            True if all fields pass checks.
        """
        all_ok = True
        
        # Density must be non-negative
        all_ok &= self.check_field(fabric.rho, "rho", tick, non_negative=True)
        
        # Energy fields must be non-negative
        all_ok &= self.check_field(fabric.E_heat, "E_heat", tick, non_negative=True)
        all_ok &= self.check_field(fabric.E_rad, "E_rad", tick, non_negative=True)
        
        # Momentum can be any sign
        all_ok &= self.check_field(fabric.px, "px", tick)
        all_ok &= self.check_field(fabric.py, "py", tick)
        
        # BH mass must be non-negative
        all_ok &= self.check_field(fabric.BH_mass, "BH_mass", tick, non_negative=True)
        
        # Influence can be any sign
        all_ok &= self.check_field(fabric.influence, "influence", tick)
        
        return all_ok

    @eidosian()
    def compute_field_stats(self, field: np.ndarray, name: str) -> FieldStats:
        """Compute statistics for a field.
        
        Args:
            field: Field array.
            name: Field name.
            
        Returns:
            FieldStats instance.
        """
        nan_count = int(np.sum(np.isnan(field)))
        inf_count = int(np.sum(np.isinf(field)))
        
        # Compute stats on finite values only
        finite_mask = np.isfinite(field)
        if np.any(finite_mask):
            finite_vals = field[finite_mask]
            min_val = float(np.min(finite_vals))
            max_val = float(np.max(finite_vals))
            mean_val = float(np.mean(finite_vals))
            std_val = float(np.std(finite_vals))
            negative_count = int(np.sum(finite_vals < 0))
        else:
            min_val = max_val = mean_val = std_val = float('nan')
            negative_count = 0
        
        return FieldStats(
            name=name,
            min_val=min_val,
            max_val=max_val,
            mean_val=mean_val,
            std_val=std_val,
            nan_count=nan_count,
            inf_count=inf_count,
            negative_count=negative_count
        )

    @eidosian()
    def take_snapshot(self, fabric: 'Fabric', tick: int) -> DiagnosticSnapshot:
        """Take a diagnostic snapshot of the simulation state.
        
        Args:
            fabric: Fabric instance.
            tick: Current tick.
            
        Returns:
            DiagnosticSnapshot instance.
        """
        field_stats = {}
        
        fields = {
            'rho': fabric.rho,
            'px': fabric.px,
            'py': fabric.py,
            'E_heat': fabric.E_heat,
            'E_rad': fabric.E_rad,
            'influence': fabric.influence,
            'BH_mass': fabric.BH_mass,
        }
        
        for name, field in fields.items():
            field_stats[name] = self.compute_field_stats(field, name)
        
        # Compute totals
        total_mass = float(fabric.rho.sum() + fabric.BH_mass.sum())
        total_energy = float(fabric.E_heat.sum() + fabric.E_rad.sum())
        
        # Add kinetic energy (where finite)
        rho = fabric.rho
        px = fabric.px
        py = fabric.py
        mask = (rho > 1e-12) & np.isfinite(rho) & np.isfinite(px) & np.isfinite(py)
        if np.any(mask):
            ke = (px[mask]**2 + py[mask]**2) / (2.0 * rho[mask])
            ke = np.clip(ke, 0, 1e100)
            total_energy += float(np.sum(ke))
        
        return DiagnosticSnapshot(
            tick=tick,
            field_stats=field_stats,
            total_mass=total_mass,
            total_energy=total_energy,
            issues=list(self.issues)
        )

    def _report_issue(
        self,
        message: str,
        field_name: str,
        tick: int,
        locations: List[Tuple[int, int]],
        values: Optional[List[float]]
    ) -> None:
        """Report an issue.
        
        Args:
            message: Issue description.
            field_name: Field where issue occurred.
            tick: Tick when issue occurred.
            locations: Grid locations.
            values: Values at locations.
        """
        if len(self.issues) >= self.max_issues:
            return
        
        issue_str = f"[tick {tick}] {message}"
        self.issues.append(issue_str)
        
        if self.fail_fast:
            raise InvariantViolation(
                message=message,
                field_name=field_name,
                tick=tick,
                locations=locations,
                values=values
            )
        elif self.warn_on_issue:
            warnings.warn(issue_str)

    @eidosian()
    def clear_issues(self) -> None:
        """Clear recorded issues."""
        self.issues.clear()


@eidosian()
def clamp_non_negative(
    field: np.ndarray,
    name: str = "field",
    record_loss: bool = True
) -> float:
    """Clamp negative values to zero and return total mass lost.
    
    Args:
        field: Field to clamp (modified in place).
        name: Field name for diagnostics.
        record_loss: If True, compute and return mass lost.
        
    Returns:
        Total mass lost to clamping (if record_loss is True).
    """
    neg_mask = field < 0
    if not np.any(neg_mask):
        return 0.0
    
    loss = 0.0
    if record_loss:
        loss = float(-np.sum(field[neg_mask]))
    
    field[neg_mask] = 0.0
    return loss


@eidosian()
def clamp_to_finite(field: np.ndarray, default: float = 0.0) -> int:
    """Replace NaN and Inf with default value.
    
    Args:
        field: Field to clamp (modified in place).
        default: Default value for non-finite values.
        
    Returns:
        Count of values replaced.
    """
    non_finite = ~np.isfinite(field)
    count = int(np.sum(non_finite))
    if count > 0:
        field[non_finite] = default
    return count


@eidosian()
def check_stability_bounds(
    max_velocity: float,
    diffusivity: float,
    dt: float,
    dx: float
) -> Tuple[bool, str]:
    """Check if parameters satisfy stability bounds.
    
    Args:
        max_velocity: Maximum velocity in simulation.
        diffusivity: Diffusion coefficient.
        dt: Timestep.
        dx: Grid spacing.
        
    Returns:
        Tuple of (is_stable, message).
    """
    issues = []
    is_stable = True
    
    # CFL condition for advection: |v|*dt/dx <= 1
    cfl_advection = max_velocity * dt / dx
    if cfl_advection > 1.0:
        is_stable = False
        issues.append(f"Advection CFL violated: {cfl_advection:.3f} > 1.0")
    
    # Diffusion stability: D*dt/dx^2 <= 0.5 (for 2D explicit)
    cfl_diffusion = diffusivity * dt / (dx * dx)
    if cfl_diffusion > 0.5:
        is_stable = False
        issues.append(f"Diffusion stability violated: {cfl_diffusion:.3f} > 0.5")
    
    if is_stable:
        return True, "All stability bounds satisfied"
    else:
        return False, "; ".join(issues)
