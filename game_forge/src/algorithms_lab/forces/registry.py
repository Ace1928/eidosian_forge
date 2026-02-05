"""Force registry and packing utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from algorithms_lab.core import ensure_f32
from algorithms_lab.forces.base import ForceDefinition, ForceType, DEFAULT_FORCE_PARAMS


@dataclass
class ForcePack:
    """Packed force arrays suitable for Numba kernels."""

    matrices: np.ndarray
    force_types: np.ndarray
    min_radius: np.ndarray
    max_radius: np.ndarray
    strength: np.ndarray
    params: np.ndarray


@dataclass
class ForceRegistry:
    """Registry for multiple force definitions.

    The registry manages per-force interaction matrices and produces
    packed arrays for Numba-accelerated kernels.
    """

    num_types: int
    forces: List[ForceDefinition] = field(default_factory=list)
    _skip_defaults: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if not self.forces and not self._skip_defaults:
            self._init_defaults()

    def _init_defaults(self) -> None:
        """Seed the registry with a curated set of forces."""

        mat_linear = np.random.uniform(-1.0, 1.0, (self.num_types, self.num_types)).astype(np.float32)
        self.add_force(
            ForceDefinition(
                name="Particle Life",
                force_type=ForceType.LINEAR,
                matrix=mat_linear,
                min_radius=0.02,
                max_radius=0.12,
                strength=1.0,
                params=DEFAULT_FORCE_PARAMS[ForceType.LINEAR].copy(),
            )
        )

        mat_gravity = np.zeros((self.num_types, self.num_types), dtype=np.float32)
        self.add_force(
            ForceDefinition(
                name="Gravity",
                force_type=ForceType.INVERSE_SQUARE,
                matrix=mat_gravity,
                min_radius=0.0,
                max_radius=0.5,
                strength=0.5,
                params=DEFAULT_FORCE_PARAMS[ForceType.INVERSE_SQUARE].copy(),
                enabled=False,
            )
        )

        mat_yukawa = np.zeros((self.num_types, self.num_types), dtype=np.float32)
        self.add_force(
            ForceDefinition(
                name="Yukawa",
                force_type=ForceType.YUKAWA,
                matrix=mat_yukawa,
                min_radius=0.0,
                max_radius=0.4,
                strength=1.0,
                params=DEFAULT_FORCE_PARAMS[ForceType.YUKAWA].copy(),
                enabled=False,
            )
        )

        mat_lj = np.zeros((self.num_types, self.num_types), dtype=np.float32)
        self.add_force(
            ForceDefinition(
                name="Lennard-Jones",
                force_type=ForceType.LENNARD_JONES,
                matrix=mat_lj,
                min_radius=0.0,
                max_radius=0.3,
                strength=1.0,
                params=DEFAULT_FORCE_PARAMS[ForceType.LENNARD_JONES].copy(),
                enabled=False,
            )
        )

        mat_morse = np.zeros((self.num_types, self.num_types), dtype=np.float32)
        self.add_force(
            ForceDefinition(
                name="Morse Bond",
                force_type=ForceType.MORSE,
                matrix=mat_morse,
                min_radius=0.0,
                max_radius=0.3,
                strength=1.0,
                params=DEFAULT_FORCE_PARAMS[ForceType.MORSE].copy(),
                enabled=False,
            )
        )

    def add_force(self, force: ForceDefinition) -> int:
        """Add a force and return its index."""

        if force.matrix.shape != (self.num_types, self.num_types):
            raise ValueError("Force matrix shape does not match registry num_types")
        force.ensure_default_params()
        self.forces.append(force)
        return len(self.forces) - 1

    def remove_force(self, index: int) -> Optional[ForceDefinition]:
        """Remove and return a force definition."""

        if 0 <= index < len(self.forces):
            return self.forces.pop(index)
        return None

    def get_force(self, name: str) -> Optional[ForceDefinition]:
        """Retrieve a force definition by name."""

        for force in self.forces:
            if force.name == name:
                return force
        return None

    def enable_force(self, name: str, enabled: bool = True) -> None:
        """Enable or disable a force definition by name."""

        force = self.get_force(name)
        if force is not None:
            force.enabled = enabled

    def get_active_forces(self) -> List[ForceDefinition]:
        """Return the active (enabled) forces."""

        return [force for force in self.forces if force.enabled]

    def set_num_types(self, n_types: int) -> None:
        """Resize force matrices to a new number of species."""

        if n_types == self.num_types:
            return
        old_n = self.num_types
        self.num_types = n_types

        for force in self.forces:
            old_matrix = force.matrix
            new_matrix = np.zeros((n_types, n_types), dtype=np.float32)
            min_n = min(old_n, n_types)
            new_matrix[:min_n, :min_n] = old_matrix[:min_n, :min_n]
            if n_types > old_n:
                new_matrix[old_n:, :] = np.random.uniform(-1.0, 1.0, (n_types - old_n, n_types))
                new_matrix[:, old_n:] = np.random.uniform(-1.0, 1.0, (n_types, n_types - old_n))
            force.matrix = new_matrix

    def pack(self) -> ForcePack:
        """Pack active forces into contiguous arrays."""

        active = self.get_active_forces()
        n_forces = len(active)
        if n_forces == 0:
            empty = np.zeros((0, self.num_types, self.num_types), dtype=np.float32)
            return ForcePack(
                matrices=empty,
                force_types=np.zeros(0, dtype=np.int32),
                min_radius=np.zeros(0, dtype=np.float32),
                max_radius=np.zeros(0, dtype=np.float32),
                strength=np.zeros(0, dtype=np.float32),
                params=np.zeros((0, 4), dtype=np.float32),
            )

        matrices = np.zeros((n_forces, self.num_types, self.num_types), dtype=np.float32)
        force_types = np.zeros(n_forces, dtype=np.int32)
        min_radius = np.zeros(n_forces, dtype=np.float32)
        max_radius = np.zeros(n_forces, dtype=np.float32)
        strength = np.zeros(n_forces, dtype=np.float32)
        params = np.zeros((n_forces, 4), dtype=np.float32)

        for idx, force in enumerate(active):
            matrices[idx] = ensure_f32(force.matrix)
            force_types[idx] = int(force.force_type)
            min_radius[idx] = float(force.min_radius)
            max_radius[idx] = float(force.max_radius)
            strength[idx] = float(force.strength)
            if force.params.size < 4:
                pad = np.zeros(4 - force.params.size, dtype=np.float32)
                params[idx] = np.concatenate([force.params, pad])
            else:
                params[idx] = ensure_f32(force.params[:4])

        return ForcePack(
            matrices=matrices,
            force_types=force_types,
            min_radius=min_radius,
            max_radius=max_radius,
            strength=strength,
            params=params,
        )

    def randomize_all(self, low: float = -1.0, high: float = 1.0) -> None:
        """Randomize all interaction matrices."""

        for force in self.forces:
            force.randomize_matrix(low, high)

    def clear_all(self) -> None:
        """Clear all interaction matrices."""

        for force in self.forces:
            force.matrix.fill(0.0)

    def get_max_radius(self) -> float:
        """Return the maximum interaction radius among active forces."""

        active = self.get_active_forces()
        if not active:
            return 0.0
        return max(float(force.max_radius) for force in active)

    def to_dict(self) -> Dict[str, object]:
        """Serialize registry to a dictionary."""

        return {
            "num_types": int(self.num_types),
            "forces": [
                {
                    "name": force.name,
                    "force_type": int(force.force_type),
                    "matrix": force.matrix.tolist(),
                    "min_radius": float(force.min_radius),
                    "max_radius": float(force.max_radius),
                    "strength": float(force.strength),
                    "params": force.params.tolist(),
                    "enabled": bool(force.enabled),
                    "symmetric": bool(force.symmetric),
                }
                for force in self.forces
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ForceRegistry":
        """Deserialize registry from a dictionary."""

        num_types = int(data.get("num_types", 0))
        registry = cls(num_types=num_types, forces=[], _skip_defaults=True)
        for force_data in data.get("forces", []):
            if not isinstance(force_data, dict):
                continue
            registry.add_force(
                ForceDefinition(
                    name=str(force_data.get("name", "Force")),
                    force_type=ForceType(int(force_data.get("force_type", 0))),
                    matrix=np.array(force_data.get("matrix", []), dtype=np.float32),
                    min_radius=float(force_data.get("min_radius", 0.0)),
                    max_radius=float(force_data.get("max_radius", 0.0)),
                    strength=float(force_data.get("strength", 1.0)),
                    params=np.array(force_data.get("params", []), dtype=np.float32),
                    enabled=bool(force_data.get("enabled", True)),
                    symmetric=bool(force_data.get("symmetric", True)),
                )
            )
        return registry
