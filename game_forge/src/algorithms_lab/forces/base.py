"""Force definitions and packed configuration for Algorithms Lab."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict

import numpy as np

from algorithms_lab.core import ensure_f32


class ForceType(IntEnum):
    """Force kernel identifiers for particle interactions."""

    LINEAR = 0
    INVERSE = 1
    INVERSE_SQUARE = 2
    INVERSE_CUBE = 3
    EXPONENTIAL = 4
    GAUSSIAN = 5
    YUKAWA = 6
    LENNARD_JONES = 7
    MORSE = 8


DEFAULT_FORCE_PARAMS: Dict[ForceType, np.ndarray] = {
    ForceType.LINEAR: np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    ForceType.INVERSE: np.array([0.01, 0.0, 0.0, 0.0], dtype=np.float32),
    ForceType.INVERSE_SQUARE: np.array([0.01, 0.0, 0.0, 0.0], dtype=np.float32),
    ForceType.INVERSE_CUBE: np.array([0.01, 0.0, 0.0, 0.0], dtype=np.float32),
    ForceType.EXPONENTIAL: np.array([0.0, 0.1, 0.0, 0.0], dtype=np.float32),
    ForceType.GAUSSIAN: np.array([0.0, 0.1, 0.0, 0.0], dtype=np.float32),
    ForceType.YUKAWA: np.array([0.01, 0.1, 0.0, 0.0], dtype=np.float32),
    ForceType.LENNARD_JONES: np.array([0.005, 0.05, 0.0, 0.0], dtype=np.float32),
    ForceType.MORSE: np.array([0.0, 0.1, 5.0, 0.0], dtype=np.float32),
}


@dataclass
class ForceDefinition:
    """Definition of a force interaction family.

    Attributes:
        name: Human-readable identifier.
        force_type: Kernel/potential shape identifier.
        matrix: (T x T) interaction strengths between species.
        min_radius: Repulsion core radius (used by the linear/particle-life kernel).
        max_radius: Maximum interaction radius (cutoff).
        strength: Global scalar multiplier for this force.
    params: Extra kernel parameters (length 4).
    mass_weighted: Whether to multiply force magnitude by mass_i * mass_j.
        enabled: Whether this force is active.
        symmetric: Whether matrix[i,j] == matrix[j,i].
    """

    name: str
    force_type: ForceType
    matrix: np.ndarray
    min_radius: float = 0.02
    max_radius: float = 0.2
    strength: float = 1.0
    params: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    enabled: bool = True
    symmetric: bool = True
    mass_weighted: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.matrix, np.ndarray):
            self.matrix = np.array(self.matrix, dtype=np.float32)
        self.matrix = ensure_f32(self.matrix)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("ForceDefinition matrix must be square (T x T)")
        if not isinstance(self.params, np.ndarray):
            self.params = np.array(self.params, dtype=np.float32)
        self.params = ensure_f32(self.params)
        if self.params.size < 4:
            pad = np.zeros(4 - self.params.size, dtype=np.float32)
            self.params = np.concatenate([self.params, pad])
        elif self.params.size > 4:
            self.params = self.params[:4].copy()

    def get_interaction(self, type_i: int, type_j: int) -> float:
        """Return the interaction coefficient for a species pair."""

        return float(self.matrix[type_i, type_j])

    def set_interaction(self, type_i: int, type_j: int, value: float, symmetrize: bool = True) -> None:
        """Set the interaction coefficient for a species pair."""

        self.matrix[type_i, type_j] = value
        if symmetrize and self.symmetric and type_i != type_j:
            self.matrix[type_j, type_i] = value

    def randomize_matrix(self, low: float = -1.0, high: float = 1.0) -> None:
        """Randomize the interaction matrix in-place."""

        n = self.matrix.shape[0]
        self.matrix = np.random.uniform(low, high, (n, n)).astype(np.float32)
        if self.symmetric:
            self.matrix = (self.matrix + self.matrix.T) / 2.0

    def ensure_default_params(self) -> None:
        """Fill params with force-type defaults when empty."""

        defaults = DEFAULT_FORCE_PARAMS[self.force_type]
        if np.allclose(self.params, 0.0):
            self.params = defaults.copy()

    def copy(self) -> "ForceDefinition":
        """Return a deep copy of this force definition."""

        return ForceDefinition(
            name=self.name,
            force_type=self.force_type,
            matrix=self.matrix.copy(),
            min_radius=float(self.min_radius),
            max_radius=float(self.max_radius),
            strength=float(self.strength),
            params=self.params.copy(),
            enabled=bool(self.enabled),
            symmetric=bool(self.symmetric),
            mass_weighted=bool(self.mass_weighted),
        )
