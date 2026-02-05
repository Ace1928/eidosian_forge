"""Extended Position-Based Dynamics (XPBD) solver for fluids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from algorithms_lab.core import Domain, ensure_f32
from algorithms_lab.kernels import poly6, spiky_grad
from algorithms_lab.neighbor_list import NeighborList


@dataclass
class XPBFState:
    """Mutable XPBD fluid state."""

    positions: NDArray[np.float32]
    velocities: NDArray[np.float32]
    masses: NDArray[np.float32]


class XPBFSolver:
    """XPBD-style compliant fluid solver.

    Compliance allows softer constraints compared to strict PBF, reducing
    numerical stiffness while maintaining stability.
    """

    def __init__(
        self,
        domain: Domain,
        h: float,
        rest_density: float = 1000.0,
        dt: float = 0.01,
        iterations: int = 4,
        gravity: float | None = None,
        compliance: float = 0.0,
        s_corr_k: float = 0.001,
        s_corr_n: int = 4,
        s_corr_q: float = 0.3,
        neighbor_backend: str = "auto",
    ) -> None:
        if h <= 0:
            raise ValueError("h must be positive")
        if iterations < 1:
            raise ValueError("iterations must be positive")
        if compliance < 0:
            raise ValueError("compliance must be non-negative")
        self.domain = domain
        self.h = float(h)
        self.rest_density = float(rest_density)
        self.dt = float(dt)
        self.iterations = int(iterations)
        self.gravity = gravity
        self.compliance = float(compliance)
        self.s_corr_k = float(s_corr_k)
        self.s_corr_n = int(s_corr_n)
        self.s_corr_q = float(s_corr_q)
        self._neighbor_list = NeighborList(domain, cutoff=h, skin=0.0, backend=neighbor_backend)

    def step(self, state: XPBFState) -> XPBFState:
        """Advance the XPBD fluid simulation by one step."""

        pos = ensure_f32(state.positions)
        vel = ensure_f32(state.velocities)
        mass = np.asarray(state.masses, dtype=np.float32)
        n = pos.shape[0]

        if self.gravity is not None:
            gravity_vec = np.zeros(self.domain.dims, dtype=np.float32)
            gravity_vec[-1] = -self.gravity
            vel = vel + gravity_vec * self.dt

        predicted = pos + vel * self.dt
        predicted = self.domain.apply_boundary(predicted)

        neighbor_data = self._neighbor_list.get(predicted)
        counts = np.diff(neighbor_data.offsets)
        idx_i = np.repeat(np.arange(n, dtype=np.int32), counts)
        idx_j = neighbor_data.neighbors

        lambdas = np.zeros(n, dtype=np.float32)
        alpha = self.compliance / (self.dt * self.dt)

        for _ in range(self.iterations):
            delta = predicted[idx_j] - predicted[idx_i]
            delta = self.domain.minimal_image(delta)
            r = np.linalg.norm(delta, axis=1)

            w = poly6(r, self.h, self.domain.dims)
            density = np.bincount(
                idx_i, weights=mass[idx_j] * w, minlength=n
            ).astype(np.float32)
            density += mass * poly6(
                np.zeros(n, dtype=np.float32), self.h, self.domain.dims
            )

            constraint = density / self.rest_density - 1.0

            grad = spiky_grad(delta, r, self.h, self.domain.dims)
            grad_ij = -mass[idx_j][:, None] * grad / self.rest_density

            grad_sum = np.zeros_like(predicted)
            for axis in range(self.domain.dims):
                grad_sum[:, axis] = np.bincount(
                    idx_i, weights=grad_ij[:, axis], minlength=n
                ).astype(np.float32)
            grad_sq = np.bincount(
                idx_i, weights=np.einsum("ij,ij->i", grad_ij, grad_ij), minlength=n
            ).astype(np.float32)

            denom = grad_sq + np.einsum("ij,ij->i", grad_sum, grad_sum) + alpha + 1e-6
            lambdas = (-constraint - alpha * lambdas) / denom

            w_q = poly6(
                np.array([self.s_corr_q * self.h], dtype=np.float32),
                self.h,
                self.domain.dims,
            )[0]
            s_corr = -self.s_corr_k * (w / w_q) ** self.s_corr_n

            scale = (lambdas[idx_i] + lambdas[idx_j] + s_corr)[:, None]
            delta_p = scale * grad / self.rest_density
            corr = np.zeros_like(predicted)
            for axis in range(self.domain.dims):
                corr[:, axis] = np.bincount(
                    idx_i, weights=delta_p[:, axis], minlength=n
                ).astype(np.float32)
            predicted += corr
            predicted = self.domain.apply_boundary(predicted)

        vel = (predicted - pos) / self.dt
        return XPBFState(positions=predicted, velocities=vel, masses=mass)
