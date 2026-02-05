"""
Eidosian PyParticles V6 - Force Base Types

Core abstractions for the force system including dropoff functions,
force definitions, and the kernel interface.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Optional
import numpy as np
from numba import njit


class DropoffType(IntEnum):
    """Distance falloff functions for force calculations."""
    NONE = 0          # Constant force (no falloff)
    LINEAR = 1        # F = k * (1 - r/r_max)
    QUADRATIC = 2     # F = k * (1 - r/r_max)²
    INVERSE = 3       # F = k / r
    INVERSE_SQUARE = 4  # F = k / r²
    INVERSE_CUBE = 5  # F = k / r³
    EXPONENTIAL = 6   # F = k * exp(-r/λ)
    GAUSSIAN = 7      # F = k * exp(-r²/2σ²)
    YUKAWA = 8        # F = k * exp(-r/λ) / r
    LENNARD_JONES = 9  # F = k * [(σ/r)¹² - (σ/r)⁶]
    MORSE = 10        # F = k * [exp(-2a(r-r₀)) - exp(-a(r-r₀))]


# Numba-compatible dropoff implementations
@njit(fastmath=True, cache=True)
def dropoff_none(r: float, r_max: float, params: np.ndarray) -> float:
    """Constant force, no distance falloff."""
    if r >= r_max:
        return 0.0
    return 1.0


@njit(fastmath=True, cache=True)
def dropoff_linear(r: float, r_max: float, params: np.ndarray) -> float:
    """Linear falloff: F = 1 - r/r_max."""
    if r >= r_max:
        return 0.0
    return 1.0 - r / r_max


@njit(fastmath=True, cache=True)
def dropoff_quadratic(r: float, r_max: float, params: np.ndarray) -> float:
    """Quadratic falloff: F = (1 - r/r_max)²."""
    if r >= r_max:
        return 0.0
    t = 1.0 - r / r_max
    return t * t


@njit(fastmath=True, cache=True)
def dropoff_inverse(r: float, r_max: float, params: np.ndarray) -> float:
    """Inverse falloff: F = 1/r with softening."""
    if r >= r_max:
        return 0.0
    softening = params[0] if len(params) > 0 else 0.01
    return 1.0 / (r + softening)


@njit(fastmath=True, cache=True)
def dropoff_inverse_square(r: float, r_max: float, params: np.ndarray) -> float:
    """Inverse square falloff: F = 1/r² with softening."""
    if r >= r_max:
        return 0.0
    softening = params[0] if len(params) > 0 else 0.01
    denom = r + softening
    return 1.0 / (denom * denom)


@njit(fastmath=True, cache=True)
def dropoff_inverse_cube(r: float, r_max: float, params: np.ndarray) -> float:
    """Inverse cube falloff: F = 1/r³ with softening."""
    if r >= r_max:
        return 0.0
    softening = params[0] if len(params) > 0 else 0.01
    denom = r + softening
    return 1.0 / (denom * denom * denom)


@njit(fastmath=True, cache=True)
def dropoff_exponential(r: float, r_max: float, params: np.ndarray) -> float:
    """Exponential falloff: F = exp(-r/λ)."""
    if r >= r_max:
        return 0.0
    decay_length = params[0] if len(params) > 0 else 0.1
    return np.exp(-r / decay_length)


@njit(fastmath=True, cache=True)
def dropoff_gaussian(r: float, r_max: float, params: np.ndarray) -> float:
    """Gaussian falloff: F = exp(-r²/2σ²)."""
    if r >= r_max:
        return 0.0
    sigma = params[0] if len(params) > 0 else 0.1
    return np.exp(-r * r / (2.0 * sigma * sigma))


@njit(fastmath=True, cache=True)
def dropoff_yukawa(r: float, r_max: float, params: np.ndarray) -> float:
    """Yukawa (screened Coulomb) falloff: F = exp(-r/λ) / r."""
    if r >= r_max:
        return 0.0
    decay_length = params[0] if len(params) > 0 else 0.1
    softening = params[1] if len(params) > 1 else 0.01
    return np.exp(-r / decay_length) / (r + softening)


@njit(fastmath=True, cache=True)
def dropoff_lennard_jones(r: float, r_max: float, params: np.ndarray) -> float:
    """
    Lennard-Jones 6-12 potential derivative.
    F = 24ε/r * [2(σ/r)¹² - (σ/r)⁶]
    Simplified: return the radial component, scale by strength externally.
    """
    if r >= r_max:
        return 0.0
    sigma = params[0] if len(params) > 0 else 0.05
    softening = params[1] if len(params) > 1 else 0.01
    r_safe = r + softening
    ratio = sigma / r_safe
    r6 = ratio ** 6
    r12 = r6 * r6
    # Force is derivative of potential, attractive at medium range, repulsive close
    return 2.0 * r12 - r6


@njit(fastmath=True, cache=True)
def dropoff_morse(r: float, r_max: float, params: np.ndarray) -> float:
    """
    Morse potential derivative.
    F = 2a * D * [exp(-2a(r-r₀)) - exp(-a(r-r₀))]
    Returns radial force component.
    """
    if r >= r_max:
        return 0.0
    r0 = params[0] if len(params) > 0 else 0.1   # Equilibrium distance
    a = params[1] if len(params) > 1 else 5.0    # Well width
    
    dr = r - r0
    exp1 = np.exp(-a * dr)
    exp2 = np.exp(-2.0 * a * dr)
    # Derivative gives force (positive = repulsive, negative = attractive)
    return exp2 - exp1


# Map dropoff types to functions (for use in non-JIT code)
DROPOFF_FUNCTIONS = {
    DropoffType.NONE: dropoff_none,
    DropoffType.LINEAR: dropoff_linear,
    DropoffType.QUADRATIC: dropoff_quadratic,
    DropoffType.INVERSE: dropoff_inverse,
    DropoffType.INVERSE_SQUARE: dropoff_inverse_square,
    DropoffType.INVERSE_CUBE: dropoff_inverse_cube,
    DropoffType.EXPONENTIAL: dropoff_exponential,
    DropoffType.GAUSSIAN: dropoff_gaussian,
    DropoffType.YUKAWA: dropoff_yukawa,
    DropoffType.LENNARD_JONES: dropoff_lennard_jones,
    DropoffType.MORSE: dropoff_morse,
}


@dataclass
class ForceDefinition:
    """
    Complete definition of a force interaction.
    
    Attributes:
        name: Human-readable identifier
        dropoff: Distance falloff function type
        matrix: (T x T) interaction strength matrix between species
        max_radius: Maximum interaction distance
        min_radius: Minimum distance (for repulsion zone)
        strength: Global strength multiplier
        params: Extra parameters for dropoff function (softening, decay length, etc.)
        enabled: Whether this force is active
        symmetric: Whether matrix[i,j] == matrix[j,i] (can optimize)
    """
    name: str
    dropoff: DropoffType
    matrix: np.ndarray  # (num_types, num_types) float32
    max_radius: float = 0.2
    min_radius: float = 0.02
    strength: float = 1.0
    params: np.ndarray = field(default_factory=lambda: np.array([0.05], dtype=np.float32))
    enabled: bool = True
    symmetric: bool = True
    
    def __post_init__(self):
        """Ensure matrix and params are proper numpy arrays."""
        if not isinstance(self.matrix, np.ndarray):
            self.matrix = np.array(self.matrix, dtype=np.float32)
        if not isinstance(self.params, np.ndarray):
            self.params = np.array(self.params, dtype=np.float32)
    
    def get_interaction(self, type_i: int, type_j: int) -> float:
        """Get interaction strength between two species."""
        return self.matrix[type_i, type_j]
    
    def set_interaction(self, type_i: int, type_j: int, value: float, symmetrize: bool = True):
        """Set interaction strength, optionally symmetrizing."""
        self.matrix[type_i, type_j] = value
        if symmetrize and self.symmetric and type_i != type_j:
            self.matrix[type_j, type_i] = value
    
    def randomize_matrix(self, low: float = -1.0, high: float = 1.0):
        """Randomize interaction matrix."""
        n = self.matrix.shape[0]
        self.matrix = np.random.uniform(low, high, (n, n)).astype(np.float32)
        if self.symmetric:
            self.matrix = (self.matrix + self.matrix.T) / 2


class ForceKernel:
    """
    Abstract base for force computation kernels.
    
    Subclasses implement specific force laws using Numba JIT compilation
    for maximum performance.
    """
    
    def __init__(self, name: str):
        self.name = name
    
    def compute(
        self,
        pos_i: np.ndarray,
        pos_j: np.ndarray,
        type_i: int,
        type_j: int,
        params: ForceDefinition,
    ) -> tuple[float, float]:
        """
        Compute force on particle i due to particle j.
        
        Returns:
            (fx, fy): Force components in world coordinates
        """
        raise NotImplementedError("Subclasses must implement compute()")
