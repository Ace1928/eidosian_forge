"""
Fabric subsystem defines the spatial substrate and underlying storage for
field quantities and species mixtures.

Fabric is responsible for allocating numpy arrays of appropriate size
and type for the continuous scalar and vector fields used in the
simulation (density, momentum, heat, radiation, etc.). It also stores
sparse per-cell species mixtures and black hole masks. Fabric does not
implement dynamics by itself; it provides convenient accessors for
reading and writing field values and defines utility functions for
boundary handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np

from .config import EngineConfig
from .types import Cell


class _ArraySliceView:
    def __init__(self, arr: np.ndarray, count: int):
        self._arr = arr
        self._count = count

    def __len__(self) -> int:
        return self._count

    def __iter__(self):
        return iter(self._arr[:self._count])

    def __getitem__(self, idx):
        return self._arr[:self._count][idx]

    def __contains__(self, item) -> bool:
        return item in self._arr[:self._count]

    def index(self, value) -> int:
        for idx in range(self._count):
            if self._arr[idx] == value:
                return idx
        raise ValueError(f"{value} is not in view")

    def tolist(self) -> list:
        return list(self._arr[:self._count])


class Mixture:
    """Sparse mixture representation for species concentrations in a cell."""

    def __init__(self, species_ids: List[str], masses: List[float], max_k: int | None = None) -> None:
        self._array_mode = max_k is not None
        self._max_k = max_k
        self._count = 0
        self._total_mass = 0.0
        if self._array_mode:
            self._species_ids_arr = np.empty(self._max_k or 0, dtype=object)
            self._masses_arr = np.zeros(self._max_k or 0, dtype=np.float64)
            self._set_from_lists(species_ids, masses)
        else:
            self._species_ids_list = list(species_ids)
            self._masses_list = list(masses)
            self._total_mass = float(sum(self._masses_list))

    def total_mass(self) -> float:
        if self._array_mode:
            return self._total_mass
        return float(sum(self._masses_list))

    def iter_entries(self):
        if self._array_mode:
            for idx in range(self._count):
                yield self._species_ids_arr[idx], float(self._masses_arr[idx])
        else:
            yield from zip(self._species_ids_list, self._masses_list)

    def _set_from_lists(self, species_ids: List[str], masses: List[float]) -> None:
        if not self._array_mode:
            return
        max_k = self._max_k or 0
        if len(masses) < len(species_ids):
            masses = list(masses) + [0.0] * (len(species_ids) - len(masses))
        count = min(len(species_ids), len(masses), max_k)
        if count > 0:
            self._species_ids_arr[:count] = species_ids[:count]
            self._masses_arr[:count] = masses[:count]
        if count < max_k:
            self._species_ids_arr[count:] = None
            self._masses_arr[count:] = 0.0
        self._count = count
        self._total_mass = float(np.sum(self._masses_arr[:count]))

    @property
    def species_ids(self) -> List[str] | np.ndarray:
        if self._array_mode:
            return _ArraySliceView(self._species_ids_arr, self._count)
        return self._species_ids_list

    @species_ids.setter
    def species_ids(self, value: List[str] | np.ndarray) -> None:
        if self._array_mode:
            masses_value = list(self.masses)
            self._set_from_lists(list(value), masses_value)
        else:
            self._species_ids_list = list(value)
            self._total_mass = float(sum(self._masses_list))

    @property
    def masses(self) -> List[float] | np.ndarray:
        if self._array_mode:
            return self._masses_arr[:self._count]
        return self._masses_list

    @masses.setter
    def masses(self, value: List[float] | np.ndarray) -> None:
        if self._array_mode:
            self._set_from_lists(list(self.species_ids), list(value))
        else:
            self._masses_list = list(value)
            self._total_mass = float(sum(self._masses_list))

    def normalise(self, target_total: float) -> None:
        """Scale the masses so that they sum to ``target_total``.

        Useful after advection or mass redistribution when the mixture
        mass does not match the cell's density exactly. Zeros are
        preserved.
        """
        total = self.total_mass()
        if total <= 0 or target_total <= 0:
            return
        s = target_total / total
        if self._array_mode and self._masses_arr is not None:
            self._masses_arr[:self._count] *= s
            self._total_mass = target_total
        else:
            for idx in range(len(self.masses)):
                self.masses[idx] *= s

    def get_weighted_property(self, prop_table: dict[str, float], prop_name: str) -> float:
        """Compute a weighted average of a property for all species present.

        ``prop_table`` is expected to map species ids to the property in
        question. If a species is unknown, a default value of 0.0 is
        returned. This function is used to compute effective values
        (e.g. ``rho_max,eff``) for EOS calculations.
        """
        total = self.total_mass()
        if total <= 0:
            return 0.0
        accum = 0.0
        for sid, mass in zip(self.species_ids, self.masses):
            value = prop_table.get(sid, 0.0)
            accum += mass * value
        return accum / total

    def add_species_mass(self, species_id: str, mass: float, max_k: int) -> None:
        """Add a species mass to the mixture, respecting the top K limit.

        If the species already exists, its mass is incremented. If not and
        there is space, a new entry is appended. If the mixture is full,
        the species with the smallest mass is replaced if the new mass is
        larger. This keeps the mixture focused on the dominant species.
        """
        if mass <= 0:
            return
        if self._array_mode and self._species_ids_arr is not None and self._masses_arr is not None:
            for idx in range(self._count):
                if self._species_ids_arr[idx] == species_id:
                    self._masses_arr[idx] += mass
                    self._total_mass += mass
                    return
            if self._count < max_k:
                self._species_ids_arr[self._count] = species_id
                self._masses_arr[self._count] = mass
                self._count += 1
                self._total_mass += mass
            else:
                min_idx = 0
                min_mass = float(self._masses_arr[0])
                for idx in range(1, self._count):
                    mval = float(self._masses_arr[idx])
                    if mval < min_mass:
                        min_mass = mval
                        min_idx = idx
                if mass > min_mass:
                    self._species_ids_arr[min_idx] = species_id
                    self._masses_arr[min_idx] = mass
                    self._total_mass += mass - min_mass
            return
        # list mode
        for idx, sid in enumerate(self.species_ids):
            if sid == species_id:
                self.masses[idx] += mass
                return
        if len(self.species_ids) < max_k:
            self.species_ids.append(species_id)
            self.masses.append(mass)
        else:
            min_idx = 0
            min_mass = self.masses[0]
            for idx, mval in enumerate(self.masses):
                if mval < min_mass:
                    min_mass = mval
                    min_idx = idx
            if mass > min_mass:
                self.species_ids[min_idx] = species_id
                self.masses[min_idx] = mass

    def index_of(self, species_id: str) -> int:
        if self._array_mode:
            for idx in range(self._count):
                if self._species_ids_arr[idx] == species_id:
                    return idx
            return -1
        try:
            return self._species_ids_list.index(species_id)
        except ValueError:
            return -1

    def cleanup(self, eps: float, max_k: int) -> None:
        """Remove negligible masses and trim list to at most ``max_k`` entries.
        """
        if self._array_mode and self._species_ids_arr is not None and self._masses_arr is not None:
            if self._count == 0:
                return
            masses = self._masses_arr[:self._count]
            if self._count <= max_k and np.all(masses > eps):
                return
            mask = masses > eps
            if not np.any(mask):
                self._species_ids_arr[:] = None
                self._masses_arr[:] = 0.0
                self._count = 0
                self._total_mass = 0.0
                return
            ids = self._species_ids_arr[:self._count][mask]
            masses = masses[mask]
            if len(ids) > max_k:
                order = np.argsort(masses)[::-1][:max_k]
                ids = ids[order]
                masses = masses[order]
            count = len(ids)
            self._species_ids_arr[:count] = ids
            self._masses_arr[:count] = masses
            if count < (self._max_k or 0):
                self._species_ids_arr[count:] = None
                self._masses_arr[count:] = 0.0
            self._count = count
            self._total_mass = float(np.sum(self._masses_arr[:count]))
            return
        new_ids = []
        new_masses = []
        for sid, mass in zip(self.species_ids, self.masses):
            if mass > eps:
                new_ids.append(sid)
                new_masses.append(mass)
        if len(new_ids) > max_k:
            sorted_items = sorted(zip(new_ids, new_masses), key=lambda x: -x[1])[:max_k]
            new_ids, new_masses = zip(*sorted_items)
            new_ids = list(new_ids)
            new_masses = list(new_masses)
        self.species_ids = new_ids
        self.masses = new_masses


class Fabric:
    """Spatial field storage for the Stratum engine.

    The Fabric holds all continuous scalar and vector fields as numpy
    arrays. Each field is initialised according to the grid size in
    ``EngineConfig``. Mixtures are stored as a separate per-cell
    structure. Access to individual fields is provided via attributes.
    """

    def __init__(self, config: EngineConfig):
        self.cfg = config
        W, H = config.grid_w, config.grid_h
        # continuous scalar fields
        self.rho = np.zeros((W, H), dtype=np.float64)
        self.px = np.zeros((W, H), dtype=np.float64)
        self.py = np.zeros((W, H), dtype=np.float64)
        self.E_heat = np.zeros((W, H), dtype=np.float64)
        self.E_rad = np.zeros((W, H), dtype=np.float64)
        self.influence = np.zeros((W, H), dtype=np.float64)
        self.BH_mass = np.zeros((W, H), dtype=np.float64)
        self.EH_mask = np.zeros((W, H), dtype=np.float64)
        # mixtures per cell
        self.mixtures: list[list[Mixture]] = [
            [Mixture([], [], max_k=getattr(config, "mixture_top_k", None)) for _ in range(H)]
            for _ in range(W)
        ]
        self.dirty_mixtures_mask = np.zeros((W, H), dtype=bool)
        self.dirty_mixtures_list: list[tuple[int, int]] = []
        self.mix_cache_dirty_mask = np.zeros((W, H), dtype=bool)
        self.mix_cache_dirty_list: list[tuple[int, int]] = []
        self.neighbor_cache = self._build_neighbor_cache()

    def mark_mixture_dirty(self, i: int, j: int) -> None:
        """Record that a cell's mixture changed (cleanup + cache update)."""
        if not self.dirty_mixtures_mask[i, j]:
            self.dirty_mixtures_mask[i, j] = True
            self.dirty_mixtures_list.append((i, j))
        if not self.mix_cache_dirty_mask[i, j]:
            self.mix_cache_dirty_mask[i, j] = True
            self.mix_cache_dirty_list.append((i, j))

    def consume_dirty_mixtures(self) -> list[tuple[int, int]]:
        """Return dirty mixture cells and clear the tracking mask."""
        dirty = self.dirty_mixtures_list
        for i, j in dirty:
            self.dirty_mixtures_mask[i, j] = False
        self.dirty_mixtures_list = []
        return dirty

    def consume_mix_cache_dirty(self) -> list[tuple[int, int]]:
        """Return cache-dirty cells and clear the tracking mask."""
        dirty = self.mix_cache_dirty_list
        for i, j in dirty:
            self.mix_cache_dirty_mask[i, j] = False
        self.mix_cache_dirty_list = []
        return dirty

    def _build_neighbor_cache(self) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Precompute neighbor indices/validity for fast inner-loop access."""
        W, H = self.cfg.grid_w, self.cfg.grid_h
        ii = np.arange(W, dtype=np.int32)[:, None]
        jj = np.arange(H, dtype=np.int32)[None, :]
        ii_full = np.broadcast_to(ii, (W, H))
        jj_full = np.broadcast_to(jj, (W, H))

        if self.cfg.boundary == "PERIODIC":
            ip1_i = (ii_full + 1) % W
            im1_i = (ii_full - 1) % W
            jp1_j = (jj_full + 1) % H
            jm1_j = (jj_full - 1) % H
            ip1_valid = np.ones((W, H), dtype=bool)
            im1_valid = np.ones((W, H), dtype=bool)
            jp1_valid = np.ones((W, H), dtype=bool)
            jm1_valid = np.ones((W, H), dtype=bool)
        elif self.cfg.boundary == "REFLECTIVE":
            ip1_i = np.minimum(ii_full + 1, W - 1)
            im1_i = np.maximum(ii_full - 1, 0)
            jp1_j = np.minimum(jj_full + 1, H - 1)
            jm1_j = np.maximum(jj_full - 1, 0)
            ip1_valid = np.ones((W, H), dtype=bool)
            im1_valid = np.ones((W, H), dtype=bool)
            jp1_valid = np.ones((W, H), dtype=bool)
            jm1_valid = np.ones((W, H), dtype=bool)
        else:
            ip1_valid = np.broadcast_to((ii + 1) < W, (W, H))
            im1_valid = np.broadcast_to((ii - 1) >= 0, (W, H))
            jp1_valid = np.broadcast_to((jj + 1) < H, (W, H))
            jm1_valid = np.broadcast_to((jj - 1) >= 0, (W, H))
            ip1_i = np.where(ip1_valid, ii_full + 1, ii_full)
            im1_i = np.where(im1_valid, ii_full - 1, ii_full)
            jp1_j = np.where(jp1_valid, jj_full + 1, jj_full)
            jm1_j = np.where(jm1_valid, jj_full - 1, jj_full)

        ip1_j = jj_full
        im1_j = jj_full
        jp1_i = ii_full
        jm1_i = ii_full

        return {
            "ip1": (ip1_i, ip1_j, ip1_valid),
            "im1": (im1_i, im1_j, im1_valid),
            "jp1": (jp1_i, jp1_j, jp1_valid),
            "jm1": (jm1_i, jm1_j, jm1_valid),
        }

    def reset_influence(self) -> None:
        self.influence.fill(0.0)

    def reset_event_horizon(self) -> None:
        self.EH_mask.fill(0.0)

    def boundary_coord(self, i: int, j: int) -> tuple[int, int]:
        """Return a valid coordinate according to the boundary mode.

        If ``boundary`` in ``EngineConfig`` is PERIODIC then indices
        wrap around. If REFLECTIVE they are clamped. If OPEN then
        indices outside the grid are returned as‑is (and callers must
        check before accessing arrays).
        """
        W, H = self.cfg.grid_w, self.cfg.grid_h
        if self.cfg.boundary == "PERIODIC":
            return i % W, j % H
        elif self.cfg.boundary == "REFLECTIVE":
            return max(0, min(i, W - 1)), max(0, min(j, H - 1))
        else:
            return i, j

    def gradient_scalar(self, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute finite difference gradient of a scalar field.

        Returns two arrays ``grad_x`` and ``grad_y`` of the same shape
        as ``field``. Uses simple central differences for interior cells
        and one‑sided differences at boundaries consistent with the
        configured boundary condition. For efficiency, numpy operations
        are used instead of explicit Python loops.
        """
        W, H = self.cfg.grid_w, self.cfg.grid_h
        grad_x = np.zeros_like(field)
        grad_y = np.zeros_like(field)
        # interior: central differences
        grad_x[1:-1, :] = (field[2:, :] - field[:-2, :]) * 0.5
        grad_y[:, 1:-1] = (field[:, 2:] - field[:, :-2]) * 0.5
        # boundaries: periodic or reflective
        if self.cfg.boundary == "PERIODIC":
            grad_x[0, :] = (field[1, :] - field[-1, :]) * 0.5
            grad_x[-1, :] = (field[0, :] - field[-2, :]) * 0.5
            grad_y[:, 0] = (field[:, 1] - field[:, -1]) * 0.5
            grad_y[:, -1] = (field[:, 0] - field[:, -2]) * 0.5
        elif self.cfg.boundary == "REFLECTIVE":
            # one sided difference at boundaries
            grad_x[0, :] = field[1, :] - field[0, :]
            grad_x[-1, :] = field[-1, :] - field[-2, :]
            grad_y[:, 0] = field[:, 1] - field[:, 0]
            grad_y[:, -1] = field[:, -1] - field[:, -2]
        else:
            # OPEN: treat outside values as zero
            grad_x[0, :] = field[1, :] - field[0, :]
            grad_x[-1, :] = -field[-1, :]
            grad_y[:, 0] = field[:, 1] - field[:, 0]
            grad_y[:, -1] = -field[:, -1]
        return grad_x, grad_y

    def divergence_vector(self, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
        """Compute divergence of a vector field given separate components.

        Uses central differences for interior cells and simple differences
        at boundaries following the boundary condition. Returns an array
        of the same shape as the input fields.
        """
        W, H = self.cfg.grid_w, self.cfg.grid_h
        div = np.zeros((W, H), dtype=vx.dtype)
        div[1:-1, :] = (vx[2:, :] - vx[:-2, :]) * 0.5
        div[:, 1:-1] += (vy[:, 2:] - vy[:, :-2]) * 0.5
        # boundaries
        if self.cfg.boundary == "PERIODIC":
            div[0, :] += (vx[1, :] - vx[-1, :]) * 0.5
            div[-1, :] += (vx[0, :] - vx[-2, :]) * 0.5
            div[:, 0] += (vy[:, 1] - vy[:, -1]) * 0.5
            div[:, -1] += (vy[:, 0] - vy[:, -2]) * 0.5
        elif self.cfg.boundary == "REFLECTIVE":
            div[0, :] += vx[1, :] - vx[0, :]
            div[-1, :] += vx[-1, :] - vx[-2, :]
            div[:, 0] += vy[:, 1] - vy[:, 0]
            div[:, -1] += vy[:, -1] - vy[:, -2]
        else:
            div[0, :] += vx[1, :] - vx[0, :]
            div[-1, :] += -vx[-1, :]
            div[:, 0] += vy[:, 1] - vy[:, 0]
            div[:, -1] += -vy[:, -1]
        return div

    def neighbors_4(self, i: int, j: int) -> list[tuple[int, int, bool]]:
        """Return the 4-connected neighbors of cell (i, j) respecting boundary mode.
        
        Returns a list of (ni, nj, valid) tuples where:
        - ni, nj are the neighbor coordinates (possibly wrapped/clamped)
        - valid is True if this neighbor should be considered for transfers
        
        For PERIODIC boundaries, all 4 neighbors are valid.
        For REFLECTIVE boundaries, edge neighbors are clamped (point to self).
        For OPEN boundaries, out-of-bounds neighbors are marked invalid.
        
        Args:
            i: X coordinate of the cell.
            j: Y coordinate of the cell.
            
        Returns:
            List of 4 tuples: [(right), (left), (up), (down)] with (ni, nj, valid).
        """
        W, H = self.cfg.grid_w, self.cfg.grid_h
        result = []
        
        # Directions: +x, -x, +y, -y
        deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        for di, dj in deltas:
            ni, nj = i + di, j + dj
            
            if self.cfg.boundary == "PERIODIC":
                # Wrap around
                result.append((ni % W, nj % H, True))
            elif self.cfg.boundary == "REFLECTIVE":
                # Clamp to edges (neighbor becomes self at boundaries)
                clamped_i = max(0, min(ni, W - 1))
                clamped_j = max(0, min(nj, H - 1))
                # Valid only if different from self (not at boundary)
                valid = (clamped_i != i or clamped_j != j)
                result.append((clamped_i, clamped_j, valid))
            else:
                # OPEN: out-of-bounds is invalid
                valid = (0 <= ni < W and 0 <= nj < H)
                if valid:
                    result.append((ni, nj, True))
                else:
                    # Return boundary index for reference but mark invalid
                    result.append((max(0, min(ni, W - 1)), max(0, min(nj, H - 1)), False))
        
        return result

    def get_neighbor_index(self, i: int, j: int, di: int, dj: int) -> tuple[int, int, bool]:
        """Get the neighbor index with boundary handling.
        
        Args:
            i: X coordinate of the cell.
            j: Y coordinate of the cell.
            di: X offset (-1, 0, or 1).
            dj: Y offset (-1, 0, or 1).
            
        Returns:
            Tuple of (ni, nj, valid) where valid indicates if transfer is allowed.
        """
        W, H = self.cfg.grid_w, self.cfg.grid_h
        ni, nj = i + di, j + dj
        
        if self.cfg.boundary == "PERIODIC":
            return (ni % W, nj % H, True)
        elif self.cfg.boundary == "REFLECTIVE":
            clamped_i = max(0, min(ni, W - 1))
            clamped_j = max(0, min(nj, H - 1))
            # For reflective, we allow transfer but it bounces back
            # Mark as valid but coords point to boundary
            return (clamped_i, clamped_j, True)
        else:
            # OPEN: only valid if in bounds
            valid = (0 <= ni < W and 0 <= nj < H)
            if valid:
                return (ni, nj, True)
            else:
                return (i, j, False)  # Return self if out of bounds
