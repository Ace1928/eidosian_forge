"""Numba-accelerated force kernels."""

from __future__ import annotations

from typing import Optional

import numpy as np

from algorithms_lab.backends import HAS_NUMBA, njit, prange
from algorithms_lab.core import Domain, WrapMode, ensure_f32
from algorithms_lab.forces.base import ForceType
from algorithms_lab.forces.registry import ForcePack, ForceRegistry


if HAS_NUMBA:
    _FORCE_LINEAR = int(ForceType.LINEAR)
    _FORCE_INVERSE = int(ForceType.INVERSE)
    _FORCE_INVERSE_SQ = int(ForceType.INVERSE_SQUARE)
    _FORCE_INVERSE_CU = int(ForceType.INVERSE_CUBE)
    _FORCE_EXPONENTIAL = int(ForceType.EXPONENTIAL)
    _FORCE_GAUSSIAN = int(ForceType.GAUSSIAN)
    _FORCE_YUKAWA = int(ForceType.YUKAWA)
    _FORCE_LJ = int(ForceType.LENNARD_JONES)
    _FORCE_MORSE = int(ForceType.MORSE)

    @njit(cache=True, fastmath=True)
    def _force_magnitude(
        force_type: int,
        dist: float,
        min_r: float,
        max_r: float,
        factor: float,
        strength: float,
        params: np.ndarray,
    ) -> float:
        if dist >= max_r:
            return 0.0
        if force_type == _FORCE_LINEAR:
            if dist < min_r and min_r > 0.0:
                force_val = (dist / min_r) - 1.0
                force_val *= 3.0
            else:
                range_len = max_r - min_r
                if range_len <= 0.0:
                    return 0.0
                numer = np.abs(2.0 * dist - range_len - 2.0 * min_r)
                peak = 1.0 - (numer / range_len)
                force_val = factor * peak
            return force_val * strength
        if force_type == _FORCE_INVERSE:
            softening = params[0]
            denom = dist + softening
            force_val = factor / denom
            force_val *= (1.0 - dist / max_r)
            return force_val * strength
        if force_type == _FORCE_INVERSE_SQ:
            softening = params[0]
            denom = dist + softening
            force_val = factor / (denom * denom)
            force_val *= (1.0 - dist / max_r)
            return force_val * strength
        if force_type == _FORCE_INVERSE_CU:
            softening = params[0]
            denom = dist + softening
            force_val = factor / (denom * denom * denom)
            force_val *= (1.0 - dist / max_r)
            return force_val * strength
        if force_type == _FORCE_EXPONENTIAL:
            decay = params[1]
            if decay <= 0.0:
                return 0.0
            return factor * np.exp(-dist / decay) * strength
        if force_type == _FORCE_GAUSSIAN:
            sigma = params[1]
            if sigma <= 0.0:
                return 0.0
            return factor * np.exp(-0.5 * (dist * dist) / (sigma * sigma)) * strength
        if force_type == _FORCE_YUKAWA:
            softening = params[0]
            decay = params[1]
            if decay <= 0.0:
                return 0.0
            r = dist + softening
            inv_r = 1.0 / r
            exp_term = np.exp(-dist / decay)
            force_val = factor * (inv_r * inv_r + inv_r / decay) * exp_term
            return force_val * strength
        if force_type == _FORCE_LJ:
            softening = params[0]
            sigma = params[1]
            if sigma <= 0.0:
                return 0.0
            r = dist + softening
            ratio = sigma / r
            r6 = ratio ** 6
            r12 = r6 * r6
            force_val = (24.0 * factor / r) * (2.0 * r12 - r6)
            return force_val * strength
        if force_type == _FORCE_MORSE:
            r0 = params[1]
            well_width = params[2]
            if well_width <= 0.0:
                return 0.0
            dr = dist - r0
            exp1 = np.exp(-well_width * dr)
            exp2 = np.exp(-2.0 * well_width * dr)
            force_val = 2.0 * well_width * factor * (exp1 - exp2)
            return force_val * strength
        return 0.0

    @njit(cache=True, fastmath=True, parallel=True)
    def accumulate_forces(
        positions: np.ndarray,
        type_ids: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        force_types: np.ndarray,
        min_radius: np.ndarray,
        max_radius: np.ndarray,
        strength: np.ndarray,
        params: np.ndarray,
        matrices: np.ndarray,
        sizes: np.ndarray,
        inv_sizes: np.ndarray,
        wrap: bool,
    ) -> np.ndarray:
        n = positions.shape[0]
        dims = positions.shape[1]
        acc = np.zeros((n, dims), dtype=np.float32)
        n_edges = rows.size
        n_forces = force_types.size
        if n_edges == 0 or n_forces == 0:
            return acc

        for e in prange(n_edges):
            i = rows[e]
            j = cols[e]
            if i == j:
                continue
            if dims == 2:
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                if wrap:
                    dx -= sizes[0] * np.round(dx * inv_sizes[0])
                    dy -= sizes[1] * np.round(dy * inv_sizes[1])
                dist2 = dx * dx + dy * dy
                if dist2 <= 0.0:
                    continue
                dist = np.sqrt(dist2)
                inv_dist = 1.0 / dist
                fx = 0.0
                fy = 0.0
                type_i = type_ids[i]
                type_j = type_ids[j]
                for f in range(n_forces):
                    factor = matrices[f, type_i, type_j]
                    if factor == 0.0:
                        continue
                    mag = _force_magnitude(
                        force_types[f],
                        dist,
                        min_radius[f],
                        max_radius[f],
                        factor,
                        strength[f],
                        params[f],
                    )
                    if mag == 0.0:
                        continue
                    fx += mag * dx * inv_dist
                    fy += mag * dy * inv_dist
                acc[i, 0] += fx
                acc[i, 1] += fy
            else:
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dz = positions[i, 2] - positions[j, 2]
                if wrap:
                    dx -= sizes[0] * np.round(dx * inv_sizes[0])
                    dy -= sizes[1] * np.round(dy * inv_sizes[1])
                    dz -= sizes[2] * np.round(dz * inv_sizes[2])
                dist2 = dx * dx + dy * dy + dz * dz
                if dist2 <= 0.0:
                    continue
                dist = np.sqrt(dist2)
                inv_dist = 1.0 / dist
                fx = 0.0
                fy = 0.0
                fz = 0.0
                type_i = type_ids[i]
                type_j = type_ids[j]
                for f in range(n_forces):
                    factor = matrices[f, type_i, type_j]
                    if factor == 0.0:
                        continue
                    mag = _force_magnitude(
                        force_types[f],
                        dist,
                        min_radius[f],
                        max_radius[f],
                        factor,
                        strength[f],
                        params[f],
                    )
                    if mag == 0.0:
                        continue
                    fx += mag * dx * inv_dist
                    fy += mag * dy * inv_dist
                    fz += mag * dz * inv_dist
                acc[i, 0] += fx
                acc[i, 1] += fy
                acc[i, 2] += fz
        return acc

    @njit(cache=True, fastmath=True, parallel=True)
    def accumulate_forces_mass(
        positions: np.ndarray,
        masses: np.ndarray,
        type_ids: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        force_types: np.ndarray,
        min_radius: np.ndarray,
        max_radius: np.ndarray,
        strength: np.ndarray,
        params: np.ndarray,
        matrices: np.ndarray,
        mass_weighted: np.ndarray,
        sizes: np.ndarray,
        inv_sizes: np.ndarray,
        wrap: bool,
    ) -> np.ndarray:
        n = positions.shape[0]
        dims = positions.shape[1]
        acc = np.zeros((n, dims), dtype=np.float32)
        n_edges = rows.size
        n_forces = force_types.size
        if n_edges == 0 or n_forces == 0:
            return acc

        for e in prange(n_edges):
            i = rows[e]
            j = cols[e]
            if i == j:
                continue
            if dims == 2:
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                if wrap:
                    dx -= sizes[0] * np.round(dx * inv_sizes[0])
                    dy -= sizes[1] * np.round(dy * inv_sizes[1])
                dist2 = dx * dx + dy * dy
                if dist2 <= 0.0:
                    continue
                dist = np.sqrt(dist2)
                inv_dist = 1.0 / dist
                fx = 0.0
                fy = 0.0
                type_i = type_ids[i]
                type_j = type_ids[j]
                mass_i = masses[i]
                mass_j = masses[j]
                for f in range(n_forces):
                    factor = matrices[f, type_i, type_j]
                    if factor == 0.0:
                        continue
                    mag = _force_magnitude(
                        force_types[f],
                        dist,
                        min_radius[f],
                        max_radius[f],
                        factor,
                        strength[f],
                        params[f],
                    )
                    if mag == 0.0:
                        continue
                    if mass_weighted[f] != 0:
                        mag *= mass_i * mass_j
                    fx += mag * dx * inv_dist
                    fy += mag * dy * inv_dist
                acc[i, 0] += fx
                acc[i, 1] += fy
            else:
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dz = positions[i, 2] - positions[j, 2]
                if wrap:
                    dx -= sizes[0] * np.round(dx * inv_sizes[0])
                    dy -= sizes[1] * np.round(dy * inv_sizes[1])
                    dz -= sizes[2] * np.round(dz * inv_sizes[2])
                dist2 = dx * dx + dy * dy + dz * dz
                if dist2 <= 0.0:
                    continue
                dist = np.sqrt(dist2)
                inv_dist = 1.0 / dist
                fx = 0.0
                fy = 0.0
                fz = 0.0
                type_i = type_ids[i]
                type_j = type_ids[j]
                mass_i = masses[i]
                mass_j = masses[j]
                for f in range(n_forces):
                    factor = matrices[f, type_i, type_j]
                    if factor == 0.0:
                        continue
                    mag = _force_magnitude(
                        force_types[f],
                        dist,
                        min_radius[f],
                        max_radius[f],
                        factor,
                        strength[f],
                        params[f],
                    )
                    if mag == 0.0:
                        continue
                    if mass_weighted[f] != 0:
                        mag *= mass_i * mass_j
                    fx += mag * dx * inv_dist
                    fy += mag * dy * inv_dist
                    fz += mag * dz * inv_dist
                acc[i, 0] += fx
                acc[i, 1] += fy
                acc[i, 2] += fz
        return acc

else:

    def accumulate_forces(*args, **kwargs):  # type: ignore
        raise ImportError("numba is required for force kernels")


def accumulate_from_registry(
    positions: np.ndarray,
    type_ids: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    registry: ForceRegistry,
    domain: Domain,
    masses: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute per-particle accelerations using a ForceRegistry."""

    pack = registry.pack()
    return accumulate_from_pack(positions, type_ids, rows, cols, pack, domain, masses=masses)


def accumulate_from_pack(
    positions: np.ndarray,
    type_ids: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    pack: ForcePack,
    domain: Domain,
    masses: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute per-particle accelerations from a packed force config."""

    pos = ensure_f32(positions)
    type_ids = np.asarray(type_ids, dtype=np.int32)
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    if pos.ndim != 2 or pos.shape[1] not in (2, 3):
        raise ValueError("positions must be of shape (N, 2) or (N, 3)")
    if rows.size != cols.size:
        raise ValueError("rows/cols must have the same length")
    if pack.mass_weighted.size > 0 and np.any(pack.mass_weighted):
        if masses is None:
            raise ValueError("masses must be provided when using mass-weighted forces")
        mass = ensure_f32(np.asarray(masses, dtype=np.float32))
        return accumulate_forces_mass(
            pos,
            mass,
            type_ids,
            rows,
            cols,
            pack.force_types,
            pack.min_radius,
            pack.max_radius,
            pack.strength,
            pack.params,
            pack.matrices,
            pack.mass_weighted,
            domain.sizes,
            domain.inv_sizes,
            domain.wrap == WrapMode.WRAP,
        )
    return accumulate_forces(
        pos,
        type_ids,
        rows,
        cols,
        pack.force_types,
        pack.min_radius,
        pack.max_radius,
        pack.strength,
        pack.params,
        pack.matrices,
        domain.sizes,
        domain.inv_sizes,
        domain.wrap == WrapMode.WRAP,
    )
