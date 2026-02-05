"""Smoothed Particle Hydrodynamics (SPH) solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from algorithms_lab.core import Domain, ensure_f32
from algorithms_lab.kernels import poly6, spiky_grad, viscosity_laplacian
from algorithms_lab.neighbor_list import NeighborList


@dataclass
class SPHState:
    """Mutable SPH state."""

    positions: NDArray[np.float32]
    velocities: NDArray[np.float32]
    masses: NDArray[np.float32]


class SPHSolver:
    """SPH solver with pressure and viscosity forces."""

    def __init__(
        self,
        domain: Domain,
        h: float,
        rest_density: float = 1000.0,
        stiffness: float = 200.0,
        viscosity: float = 0.1,
        dt: float = 0.01,
        gravity: float | None = None,
        xsph: float = 0.0,
        neighbor_backend: str = "auto",
    ) -> None:
        if h <= 0:
            raise ValueError("h must be positive")
        self.domain = domain
        self.h = float(h)
        self.rest_density = float(rest_density)
        self.stiffness = float(stiffness)
        self.viscosity = float(viscosity)
        self.dt = float(dt)
        self.gravity = gravity
        self.xsph = float(xsph)
        self._neighbor_list = NeighborList(domain, cutoff=h, skin=0.0, backend=neighbor_backend)

    def step(self, state: SPHState) -> SPHState:
        """Advance the SPH simulation by one step."""

        pos = ensure_f32(state.positions)
        vel = ensure_f32(state.velocities)
        mass = np.asarray(state.masses, dtype=np.float32)

        n = pos.shape[0]
        neighbor_data = self._neighbor_list.get(pos)
        counts = np.diff(neighbor_data.offsets)
        idx_i = np.repeat(np.arange(n, dtype=np.int32), counts)
        idx_j = neighbor_data.neighbors

        delta = pos[idx_j] - pos[idx_i]
        delta = self.domain.minimal_image(delta)
        r = np.linalg.norm(delta, axis=1)

        w = poly6(r, self.h, self.domain.dims)
        density = np.bincount(
            idx_i, weights=mass[idx_j] * w, minlength=n
        ).astype(np.float32)
        density += mass * poly6(np.zeros(n, dtype=np.float32), self.h, self.domain.dims)

        pressure = self.stiffness * (density - self.rest_density)

        grad = spiky_grad(delta, r, self.h, self.domain.dims)
        pressure_term = (
            mass[idx_j]
            * (pressure[idx_i] + pressure[idx_j])
            / (2.0 * np.maximum(density[idx_j], 1e-6))
        )
        pressure_force = -grad * pressure_term[:, None]

        lap = viscosity_laplacian(r, self.h, self.domain.dims)
        visc_term = (
            self.viscosity
            * mass[idx_j][:, None]
            * (vel[idx_j] - vel[idx_i])
            / np.maximum(density[idx_j], 1e-6)[:, None]
        )
        viscosity_force = visc_term * lap[:, None]

        force = np.zeros_like(pos)
        combined = pressure_force + viscosity_force
        for axis in range(self.domain.dims):
            force[:, axis] = np.bincount(
                idx_i, weights=combined[:, axis], minlength=n
            ).astype(np.float32)

        if self.gravity is not None:
            gravity_vec = np.zeros(self.domain.dims, dtype=np.float32)
            gravity_vec[-1] = -self.gravity
            force += gravity_vec

        accel = force / np.maximum(density, 1e-6)[:, None]
        vel = vel + accel * self.dt

        if self.xsph > 0:
            xsph_delta = (vel[idx_j] - vel[idx_i]) * (w[:, None])
            xsph_accum = np.zeros_like(pos)
            for axis in range(self.domain.dims):
                xsph_accum[:, axis] = np.bincount(
                    idx_i, weights=xsph_delta[:, axis], minlength=n
                ).astype(np.float32)
            vel += self.xsph * xsph_accum

        pos = pos + vel * self.dt
        pos = self.domain.apply_boundary(pos)

        return SPHState(positions=pos, velocities=vel, masses=mass)
