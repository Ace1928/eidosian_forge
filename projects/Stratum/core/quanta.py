"""
Quanta subsystem: event propagation and local microtick resolution.

Quanta orchestrates the delivery of signals (influence, radiation,
impulses) with finite speed and processes local cell interactions in
small microticks before propagating their finalised effects outwards.
This implementation follows the design described in the specification,
but simplifies some aspects for clarity and performance: influence
propagation uses a simple signal queue, radiation energy is injected
locally for each tick and then diffuses, and active cell selection
uses basic heuristics based on gradients and overfill.

Key Performance Features:
- Derived fields (rho_max_eff, T_field, etc.) are cached per tick
- Boundary handling is centralized through Fabric methods
- Active cell selection uses efficient numpy operations
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import EngineConfig
from .types import Vec2, clamp
from .ledger import Ledger, KINETIC_ENERGY_MAX, MOMENTUM_SQ_MAX, RHO_EPSILON, RHO_MAX_CLAMP
from .fabric import Fabric, Mixture
from .registry import SpeciesRegistry
from . import types as ttypes

# Numerical stability constants
#: Maximum energy delta to prevent downstream overflow in energy conversions
ENERGY_DELTA_MAX = 1e50


class SignalType:
    """Signal type constants for the propagation system."""
    INFLUENCE = 0
    RADIATION = 1
    IMPULSE = 2
    DISTURBANCE = 3


class Signal:
    """Lightweight signal representation used for propagation.
    
    Attributes:
        type: Signal type (INFLUENCE, RADIATION, IMPULSE, DISTURBANCE).
        emit_tick: Tick when the signal was emitted.
        origin: Grid coordinates (i, j) of the signal origin.
        speed: Propagation speed in cells per tick.
        attenuation: Attenuation factor per unit distance.
        radius: Radius of effect in cells.
        payload: Dictionary of signal-specific data.
        arrive_tick: Computed tick when signal arrives.
    """

    def __init__(self, sig_type: int, emit_tick: int, origin: Tuple[int, int], speed: float, attenuation: float, radius: int, payload: dict):
        self.type = sig_type
        self.emit_tick = emit_tick
        self.origin = origin
        self.speed = speed
        self.attenuation = attenuation
        self.radius = radius
        self.payload = payload
        self.arrive_tick = emit_tick  # will be set when enqueued


class SignalQueue:
    """Time-sorted queue for signal delivery."""

    def __init__(self):
        self.by_tick: Dict[int, List[Signal]] = {}

    def push(self, sig: Signal, current_tick: int, v_max: float) -> None:
        """Insert a signal into the queue computing its arrival time."""
        # compute travel time: distance / speed; in this simplified version
        # we assume signals deposit energy in the origin cell only on the next tick
        # For a real implementation distance to footprint radius would be used.
        # Here we treat radius = 0 and schedule for next tick.
        sig.arrive_tick = current_tick + 1
        self.by_tick.setdefault(sig.arrive_tick, []).append(sig)

    def pop_arrivals(self, tick: int) -> List[Signal]:
        arrivals = self.by_tick.pop(tick, [])
        return arrivals


@dataclass
class DerivedFields:
    """Cached derived fields computed once per tick.
    
    This cache eliminates redundant computation of mixture-weighted properties
    and derived quantities during microtick processing. All fields are computed
    once at the beginning of each tick.
    
    Attributes:
        rho_max_eff: Effective maximum density per cell (mixture-weighted).
        chi_eff: Effective EOS stiffness per cell.
        eta_eff: Effective viscosity per cell.
        opacity_eff: Effective opacity per cell.
        T_field: Temperature field (E_heat / rho).
        Z_field: Compression index field.
        grad_infl_mag: Magnitude of influence gradient.
        tick: Tick number for which this cache is valid.
    """
    rho_max_eff: np.ndarray
    chi_eff: np.ndarray
    eta_eff: np.ndarray
    opacity_eff: np.ndarray
    T_field: np.ndarray
    Z_field: np.ndarray
    grad_infl_mag: np.ndarray
    grad_infl_x: np.ndarray
    grad_infl_y: np.ndarray
    grad_P_x: np.ndarray
    grad_P_y: np.ndarray
    dom_species_idx: np.ndarray
    deg_frac: np.ndarray
    dom_rho_max: np.ndarray
    dom_chi: np.ndarray
    dom_beta: np.ndarray
    dom_lambda: np.ndarray
    dom_nu: np.ndarray
    mix_species_idx: np.ndarray
    mix_masses: np.ndarray
    species_id_list: list[str]
    fill_idx: int
    tick: int


class Quanta:
    """Event propagation and microtick scheduler.
    
    The Quanta subsystem handles:
    - Signal delivery: Influence, radiation, and impulse signals
    - Active region selection: Identifies cells requiring detailed simulation
    - Microtick resolution: Local physics updates within each tick
    - Mass transfer: Conservation-aware mass movement between cells
    
    Key performance optimization: Derived fields are computed once per tick
    and cached in a DerivedFields dataclass to avoid redundant computation.
    """

    def __init__(self, fabric: Fabric, ledger: Ledger, registry: SpeciesRegistry, materials: 'MaterialsFundamentals', config: EngineConfig):
        self.fabric = fabric
        self.ledger = ledger
        self.registry = registry
        self.materials = materials
        self.cfg = config
        self.queue = SignalQueue()
        # Simple counters for diagnostics
        self.total_microticks = 0
        # Derived fields cache (computed once per tick)
        self._derived_cache: Optional[DerivedFields] = None
        self._mix_cache_masses: Optional[np.ndarray] = None
        self._mix_cache_species_idx: Optional[np.ndarray] = None
        self._mix_cache_species_count: Optional[int] = None
        self._mix_cache_max_k: Optional[int] = None

    def _compute_derived_fields(self, tick: int) -> DerivedFields:
        """Compute and cache all derived fields for the current tick.
        
        This method computes mixture-weighted properties and derived quantities
        once per tick, avoiding redundant computation during microtick processing.
        
        Args:
            tick: Current simulation tick.
            
        Returns:
            DerivedFields dataclass containing all cached fields.
        """
        W, H = self.cfg.grid_w, self.cfg.grid_h

        mixtures = self.fabric.mixtures

        species_ids = list(self.registry.species.keys())
        species_count = len(species_ids)
        fill_idx = species_count
        sid_to_idx = {sid: idx for idx, sid in enumerate(species_ids)}

        rho_max_vals = np.zeros(species_count + 1, dtype=np.float64)
        chi_vals = np.zeros(species_count + 1, dtype=np.float64)
        eta_vals = np.zeros(species_count + 1, dtype=np.float64)
        opacity_vals = np.zeros(species_count + 1, dtype=np.float64)
        beta_vals = np.zeros(species_count + 1, dtype=np.float64)
        lambda_vals = np.zeros(species_count + 1, dtype=np.float64)
        nu_vals = np.zeros(species_count + 1, dtype=np.float64)
        for idx, sid in enumerate(species_ids):
            props = self.registry.species[sid].he_props
            rho_max_vals[idx] = props.get("HE/rho_max", 0.0)
            chi_vals[idx] = props.get("HE/chi", 0.0)
            eta_vals[idx] = props.get("HE/eta", 0.0)
            opacity_vals[idx] = props.get("HE/opacity", 0.0)
            beta_vals[idx] = props.get("HE/beta", 0.0)
            lambda_vals[idx] = props.get("HE/lambda", 0.0)
            nu_vals[idx] = props.get("HE/nu", 0.0)

        max_k = self.cfg.mixture_top_k or 0
        use_cache = (
            max_k > 0
            and self._mix_cache_masses is not None
            and self._mix_cache_species_idx is not None
            and self._mix_cache_species_count == species_count
            and self._mix_cache_max_k == max_k
        )
        if use_cache:
            mix_masses = self._mix_cache_masses
            mix_species_idx = self._mix_cache_species_idx
        else:
            mix_masses = np.zeros((W, H, max_k), dtype=np.float64)
            mix_species_idx = np.full((W, H, max_k), fill_idx, dtype=np.int32)
            self._mix_cache_masses = mix_masses
            self._mix_cache_species_idx = mix_species_idx
            self._mix_cache_species_count = species_count
            self._mix_cache_max_k = max_k

        if max_k > 0:
            get_idx = sid_to_idx.get
            dirty = self.fabric.mix_cache_dirty_list
            if not use_cache or not dirty:
                dirty_iter = None
            else:
                dirty_iter = list(dirty)
            if not use_cache or dirty_iter is None:
                for i in range(W):
                    row = mixtures[i]
                    for j in range(H):
                        mix = row[j]
                        mix_masses[i, j, :] = 0.0
                        mix_species_idx[i, j, :] = fill_idx
                        if getattr(mix, "_array_mode", False):
                            count = mix._count
                            if count <= 0:
                                continue
                            mix_masses[i, j, :count] = mix._masses_arr[:count]
                            species_arr = mix._species_ids_arr
                            for k in range(count):
                                mix_species_idx[i, j, k] = get_idx(species_arr[k], fill_idx)
                        else:
                            entries = list(mix.iter_entries())
                            if not entries:
                                continue
                            count = min(len(entries), max_k)
                            for k in range(count):
                                sid, mass = entries[k]
                                mix_masses[i, j, k] = mass
                                mix_species_idx[i, j, k] = get_idx(sid, fill_idx)
            else:
                for i, j in dirty_iter:
                    mix = mixtures[i][j]
                    mix_masses[i, j, :] = 0.0
                    mix_species_idx[i, j, :] = fill_idx
                    if getattr(mix, "_array_mode", False):
                        count = mix._count
                        if count <= 0:
                            continue
                        mix_masses[i, j, :count] = mix._masses_arr[:count]
                        species_arr = mix._species_ids_arr
                        for k in range(count):
                            mix_species_idx[i, j, k] = get_idx(species_arr[k], fill_idx)
                    else:
                        entries = list(mix.iter_entries())
                        if not entries:
                            continue
                        count = min(len(entries), max_k)
                        for k in range(count):
                            sid, mass = entries[k]
                            mix_masses[i, j, k] = mass
                            mix_species_idx[i, j, k] = get_idx(sid, fill_idx)
            if dirty:
                self.fabric.consume_mix_cache_dirty()

        totals = mix_masses.sum(axis=2)
        inv_total = np.divide(1.0, totals, out=np.zeros_like(totals), where=totals > 0.0)
        rho_max_eff = (mix_masses * rho_max_vals[mix_species_idx]).sum(axis=2) * inv_total
        chi_eff = (mix_masses * chi_vals[mix_species_idx]).sum(axis=2) * inv_total
        eta_eff = (mix_masses * eta_vals[mix_species_idx]).sum(axis=2) * inv_total
        opacity_eff = (mix_masses * opacity_vals[mix_species_idx]).sum(axis=2) * inv_total

        dom_species_idx = np.full((W, H), fill_idx, dtype=np.int32)
        if max_k > 0:
            dom_species_idx = np.where(totals > 0.0, np.argmax(mix_masses, axis=2), fill_idx).astype(np.int32)
        dom_rho_max = rho_max_vals[dom_species_idx]
        dom_chi = chi_vals[dom_species_idx]
        dom_beta = beta_vals[dom_species_idx]
        dom_lambda = lambda_vals[dom_species_idx]
        dom_nu = nu_vals[dom_species_idx]

        deg_frac = np.zeros((W, H), dtype=np.float64)
        deg_species = getattr(self.materials, "deg_species", None)
        deg_idx = sid_to_idx.get(deg_species.id, None) if deg_species else None
        if deg_idx is not None and max_k > 0:
            deg_mask = mix_species_idx == deg_idx
            deg_mass = (mix_masses * deg_mask).sum(axis=2)
            deg_frac = deg_mass * inv_total

        # Add global viscosity to eta (matching prior behavior).
        eta_eff += self.cfg.viscosity_global
        
        # Compute temperature field
        T_field = np.divide(self.fabric.E_heat, np.maximum(self.fabric.rho, 1e-12))
        
        # Compute influence gradient
        grad_infl_x, grad_infl_y = self.fabric.gradient_scalar(self.fabric.influence)
        grad_infl_mag = np.sqrt(grad_infl_x ** 2 + grad_infl_y ** 2)

        rho_max_eff_eps = np.maximum(rho_max_eff, 1e-12)
        r_ratio = np.divide(self.fabric.rho, rho_max_eff_eps)
        P_eos = chi_eff * (r_ratio ** self.cfg.eos_gamma)
        P_th = self.cfg.thermal_pressure_coeff * self.fabric.rho * T_field
        over = np.maximum(self.fabric.rho - rho_max_eff, 0.0)
        P_rep = np.where(rho_max_eff > 0.0, self.cfg.repulsion_k * (over / rho_max_eff_eps) ** self.cfg.repulsion_n, 0.0)
        P_field = P_eos + P_th + P_rep
        grad_P_x, grad_P_y = self.fabric.gradient_scalar(P_field)
        
        # Compute overfill and Z field
        overfill = np.maximum(self.fabric.rho - rho_max_eff, 0.0)
        Z_field = (
            1.0 * np.divide(self.fabric.rho, np.maximum(rho_max_eff, 1e-12))
            + 0.5 * np.log1p(T_field)
            + 0.2 * grad_infl_mag
            + 0.3 * overfill
        )
        np.minimum(Z_field, self.cfg.Z_abs_max, out=Z_field)
        
        return DerivedFields(
            rho_max_eff=rho_max_eff,
            chi_eff=chi_eff,
            eta_eff=eta_eff,
            opacity_eff=opacity_eff,
            T_field=T_field,
            Z_field=Z_field,
            grad_infl_mag=grad_infl_mag,
            grad_infl_x=grad_infl_x,
            grad_infl_y=grad_infl_y,
            grad_P_x=grad_P_x,
            grad_P_y=grad_P_y,
            dom_species_idx=dom_species_idx,
            deg_frac=deg_frac,
            dom_rho_max=dom_rho_max,
            dom_chi=dom_chi,
            dom_beta=dom_beta,
            dom_lambda=dom_lambda,
            dom_nu=dom_nu,
            mix_species_idx=mix_species_idx,
            mix_masses=mix_masses,
            species_id_list=species_ids,
            fill_idx=fill_idx,
            tick=tick
        )

    def step(self, tick: int, micro_budget: int) -> None:
        """Execute one simulation tick.
        
        This method:
        1. Delivers scheduled signals
        2. Computes and caches derived fields
        3. Selects active cells for detailed processing
        4. Runs microtick resolution on active cells
        5. Applies global operators (diffusion, etc.)
        
        Args:
            tick: Current simulation tick number.
            micro_budget: Total microtick budget for this tick.
        """
        # deliver signals for this tick
        arrivals = self.queue.pop_arrivals(tick)
        W, H = self.cfg.grid_w, self.cfg.grid_h
        # temporary accumulators
        signal_hits = np.zeros((W, H), dtype=np.int32)
        influence_add = np.zeros((W, H), dtype=np.float64)
        rad_add = np.zeros((W, H), dtype=np.float64)
        impulse_x = np.zeros((W, H), dtype=np.float64)
        impulse_y = np.zeros((W, H), dtype=np.float64)
        # accumulate
        for sig in arrivals:
            x0, y0 = sig.origin
            # For now, signals deposit in the origin cell only (simplified propagation)
            if 0 <= x0 < W and 0 <= y0 < H:
                signal_hits[x0, y0] += 1
                if sig.type == SignalType.INFLUENCE:
                    influence_add[x0, y0] += sig.payload.get("strength", 0.0)
                elif sig.type == SignalType.RADIATION:
                    rad_add[x0, y0] += sig.payload.get("energy", 0.0)
                elif sig.type == SignalType.IMPULSE:
                    dp: Vec2 = sig.payload.get("dp", Vec2(0.0, 0.0))
                    impulse_x[x0, y0] += dp.x
                    impulse_y[x0, y0] += dp.y
        # apply accumulators
        self.fabric.influence += influence_add
        self.fabric.E_rad += rad_add
        self.fabric.px += impulse_x
        self.fabric.py += impulse_y
        
        # Compute and cache derived fields for this tick
        self._derived_cache = self._compute_derived_fields(tick)
        derived = self._derived_cache

        # Keep dirty mixture tracking for cache updates; cleanup happens after microticks.
        
        # compute gradient of rho for active cell selection
        grad_rho_x, grad_rho_y = self.fabric.gradient_scalar(self.fabric.rho)
        grad_rho_mag = np.sqrt(grad_rho_x ** 2 + grad_rho_y ** 2)
        
        # Compute overfill using cached rho_max_eff
        overfill = np.maximum(self.fabric.rho - derived.rho_max_eff, 0.0)
        
        # determine active cells: top cells with high grad or overfill or high Z
        # flatten arrays to rank
        scores = grad_rho_mag + overfill + np.maximum(derived.Z_field - self.cfg.Z_fuse_min * 0.5, 0.0)
        
        # Use argpartition for O(N) top-K selection instead of O(N log N) argsort
        # When all scores are equal (e.g. uniform initial conditions), this ordering
        # is arbitrary but stable. We pick the top ``active_region_max`` cells
        # regardless of the absolute score so that microticks always run on some
        # subset of the grid even in symmetric states.
        flat_scores = scores.ravel()
        n_total = len(flat_scores)
        n_active = min(self.cfg.active_region_max, n_total)
        
        if n_active > 0 and n_total > n_active:
            # Use argpartition for O(N) selection of top K elements
            # argpartition gives us indices where the K-th element is in its final position
            # and all elements before it are smaller (unordered) and after are larger (unordered)
            # We want the LARGEST K elements, so we partition at (n_total - n_active)
            partition_idx = n_total - n_active
            partitioned_indices = np.argpartition(flat_scores, partition_idx)
            top_k_indices = partitioned_indices[partition_idx:]
            
            # Sort only the top K elements for consistent ordering (optional but good for reproducibility)
            top_k_scores = flat_scores[top_k_indices]
            sorted_within_topk = np.argsort(top_k_scores)[::-1]
            flat_indices = top_k_indices[sorted_within_topk]
        else:
            # If we want all or more cells than exist, just take all (sorted)
            flat_indices = np.argsort(flat_scores)[::-1][:n_active]
        
        active_cells: List[Tuple[int, int]] = []
        for idx in flat_indices:
            i = idx // H
            j = idx % H
            active_cells.append((i, j))
        # allocate microticks evenly for simplicity
        if len(active_cells) == 0:
            return
        micro_per_cell = max(1, micro_budget // len(active_cells))
        # main local microtick loop
        M = min(micro_per_cell, self.cfg.microtick_cap_per_region)
        batch_threshold = 4096
        if M == 1 and len(active_cells) >= batch_threshold:
            self._resolve_cells_microtick(active_cells, derived, tick)
            self.total_microticks += len(active_cells)
        else:
            for (i, j) in active_cells:
                if self.fabric.EH_mask[i, j] > 0.0:
                    continue
                self.resolve_cell(i, j, M, derived, tick)
                self.total_microticks += M
        dirty = self.fabric.consume_dirty_mixtures()
        if dirty:
            eps = self.cfg.mixture_eps_merge
            top_k = self.cfg.mixture_top_k
            for i, j in dirty:
                self.fabric.mixtures[i][j].cleanup(eps, top_k)
        if active_cells:
            self._resolve_high_energy_events(active_cells, derived, tick)
        # after microticks, diffuse heat and radiation (global operators)
        self.materials.apply_global_ops(self.fabric, self.cfg, mix_cache=(derived.mix_species_idx, derived.mix_masses))
        # reset influence for next tick
        self.fabric.reset_influence()

    def _resolve_high_energy_events(
        self,
        active_cells: list[tuple[int, int]],
        derived: DerivedFields,
        tick: int,
    ) -> None:
        """Batch filter and resolve high-energy events once per cell per tick."""
        if not active_cells:
            return
        cfg = self.cfg
        rho = self.fabric.rho
        EH_mask = self.fabric.EH_mask
        coords = np.asarray(active_cells, dtype=np.int32)
        i_all = coords[:, 0]
        j_all = coords[:, 1]
        rho_all = rho[i_all, j_all]
        dom_idx_all = derived.dom_species_idx[i_all, j_all]
        valid_mask = (
            (rho_all > 1e-12)
            & (EH_mask[i_all, j_all] <= 0.0)
            & (dom_idx_all != derived.fill_idx)
        )
        if not np.any(valid_mask):
            return
        valid_idx = np.nonzero(valid_mask)[0]
        i_idx = i_all[valid_idx]
        j_idx = j_all[valid_idx]
        rho_vals = rho_all[valid_idx]

        E_heat_vals = self.fabric.E_heat[i_idx, j_idx]
        T_vals = np.divide(E_heat_vals, np.maximum(rho_vals, 1e-12))
        Z_vals = derived.Z_field[i_idx, j_idx]
        dom_beta = derived.dom_beta[i_idx, j_idx]
        dom_chi = derived.dom_chi[i_idx, j_idx]
        dom_lambda = derived.dom_lambda[i_idx, j_idx]
        dom_nu = derived.dom_nu[i_idx, j_idx]

        px_vals = self.fabric.px[i_idx, j_idx]
        py_vals = self.fabric.py[i_idx, j_idx]
        sum_sq = px_vals * px_vals + py_vals * py_vals
        sum_sq = np.where(np.isfinite(sum_sq), sum_sq, MOMENTUM_SQ_MAX)
        np.minimum(sum_sq, MOMENTUM_SQ_MAX, out=sum_sq)
        rho_clamped = np.minimum(rho_vals, RHO_MAX_CLAMP)
        kin_energy = np.where(rho_vals > RHO_EPSILON, sum_sq / (2.0 * rho_clamped), 0.0)
        np.minimum(kin_energy, KINETIC_ENERGY_MAX, out=kin_energy)
        avail_E = 0.5 * E_heat_vals + 0.5 * kin_energy

        S_high = dom_beta + dom_chi - dom_lambda
        S_high -= cfg.stability_high_coeff * np.maximum(Z_vals - cfg.Z_fuse_min, 0.0)
        S_high -= cfg.stability_temp_coeff * np.maximum(T_vals - 0.5, 0.0)

        bh = Z_vals >= cfg.Z_bh_min
        deg = (Z_vals >= cfg.Z_deg_min) & (S_high < 0.0)
        fuse = (Z_vals >= cfg.Z_fuse_min) & (S_high < 0.0) & (dom_nu > 0.0)
        decay = (Z_vals <= cfg.Z_fuse_min) & (S_high < -0.1)

        deg_frac_vals = derived.deg_frac[i_idx, j_idx]
        gate_bh = np.clip(deg_frac_vals + 0.1, 0.0, 1.0)
        gate_deg = np.clip(1.0 - deg_frac_vals, 0.0, 1.0)
        gate_fus = np.clip(dom_nu, 0.0, 1.0)
        gate_dec = np.clip(dom_lambda, 0.0, 1.0)

        bh_act = 1.2 - np.clip(
            (Z_vals - cfg.Z_bh_min) / max(cfg.Z_abs_max - cfg.Z_bh_min, 1e-6),
            0.0,
            1.0,
        )
        E_act_bh = np.maximum(0.1, bh_act * np.exp(-0.5 * T_vals))
        deg_act = 1.0 - 0.8 * np.clip(
            (Z_vals - cfg.Z_deg_min) / max(cfg.Z_abs_max - cfg.Z_deg_min, 1e-6),
            0.0,
            1.0,
        )
        E_act_deg = np.maximum(0.05, deg_act * np.exp(-T_vals))
        fus_base = 0.5 + 0.5 * (derived.dom_rho_max[i_idx, j_idx] + dom_chi)
        fus_z_factor = np.clip(
            (Z_vals - cfg.Z_fuse_min) / max(cfg.Z_abs_max - cfg.Z_fuse_min, 1e-6),
            0.0,
            1.0,
        )
        fus_base *= np.maximum(0.1, 1.0 - fus_z_factor)
        E_act_fus = fus_base * np.exp(-T_vals)
        dec_base = 0.3 + 0.7 * dom_lambda
        dec_z_factor = np.clip(
            (cfg.Z_fuse_min - Z_vals) / max(cfg.Z_fuse_min, 1e-6),
            0.0,
            1.0,
        )
        dec_base *= np.maximum(0.05, 1.0 - dec_z_factor)
        E_act_dec = dec_base * np.exp(-T_vals)

        bh = bh & (gate_bh > 0.0) & ((avail_E >= E_act_bh) | (T_vals > 0.0))
        deg = deg & (gate_deg > 0.0) & ((avail_E >= E_act_deg) | (T_vals > 0.0))
        fuse = fuse & (gate_fus > 0.0) & ((avail_E >= E_act_fus) | (T_vals > 0.0))
        decay = decay & (gate_dec > 0.0) & ((avail_E >= E_act_dec) | (T_vals > 0.0))

        candidates = bh | deg | fuse | decay

        if not np.any(candidates):
            return
        cand_idx = np.nonzero(candidates)[0]
        for idx in cand_idx:
            i = int(i_idx[idx])
            j = int(j_idx[idx])
            dom_idx = int(dom_idx_all[valid_idx[idx]])
            if dom_idx == derived.fill_idx:
                continue
            self.materials.handle_high_energy_events(
                self.fabric,
                self.ledger,
                self.registry,
                i,
                j,
                float(Z_vals[idx]),
                float(T_vals[idx]),
                tick,
                attempt=0,
                dom_sid=derived.species_id_list[dom_idx],
                deg_frac=float(deg_frac_vals[idx]),
                dom_rho_max=float(derived.dom_rho_max[i, j]),
                dom_chi=float(dom_chi[idx]),
                dom_beta=float(dom_beta[idx]),
                dom_lambda=float(dom_lambda[idx]),
                dom_nu=float(dom_nu[idx]),
                avail_E=float(avail_E[idx]),
                E_act_bh=float(E_act_bh[idx]),
                E_act_deg=float(E_act_deg[idx]),
                E_act_fus=float(E_act_fus[idx]),
                E_act_dec=float(E_act_dec[idx]),
                gate_bh=float(gate_bh[idx]),
                gate_deg=float(gate_deg[idx]),
                gate_fus=float(gate_fus[idx]),
                gate_dec=float(gate_dec[idx]),
                bh_possible=bool(bh[idx]),
                deg_possible=bool(deg[idx]),
                fuse_possible=bool(fuse[idx]),
                decay_possible=bool(decay[idx]),
            )

    def compute_effective_rho_max(self) -> np.ndarray:
        """Compute the mixture weighted effective rho_max per cell.

        Returns a numpy array of shape (W,H). Use HE/rho_max table from
        registry. If a species is unknown in the registry, zero is
        assumed, which will result in zero effective rho_max; caller
        should treat such cells carefully.
        """
        W, H = self.cfg.grid_w, self.cfg.grid_h
        rho_max_eff = np.zeros((W, H), dtype=np.float64)
        # build a property lookup table for rho_max
        rho_max_table = {sid: s.he_props.get("HE/rho_max", 0.0) for sid, s in self.registry.species.items()}
        # compute per cell
        for i in range(W):
            for j in range(H):
                mix = self.fabric.mixtures[i][j]
                total = mix.total_mass()
                if total <= 0.0:
                    rho_max_eff[i, j] = 0.0
                    continue
                accum = 0.0
                for sid, mass in mix.iter_entries():
                    accum += mass * rho_max_table.get(sid, 0.0)
                rho_max_eff[i, j] = accum / total
        return rho_max_eff

    def resolve_cell(self, i: int, j: int, M: int, derived: DerivedFields, tick: int) -> None:
        """Perform ``M`` local microticks at cell (i,j).

        This method implements a simplified version of the detailed logic
        from the specification. It computes pressure and gravitational
        accelerations, updates momentum, performs a small advection
        substep and corrects overfill.
        
        Uses cached derived fields from DerivedFields to avoid redundant
        computation of mixture-weighted properties.
        
        Args:
            i: X coordinate of the cell.
            j: Y coordinate of the cell.
            M: Number of microticks to perform.
            derived: Cached derived fields for this tick.
            tick: Current simulation tick.
        """
        # local aliases for speed
        cfg = self.cfg
        rho_array = self.fabric.rho
        px = self.fabric.px
        py = self.fabric.py
        E_heat = self.fabric.E_heat
        E_rad = self.fabric.E_rad
        EH_mask = self.fabric.EH_mask
        neighbor_cache = self.fabric.neighbor_cache
        ip1_i_arr, ip1_j_arr, ip1_valid_arr = neighbor_cache["ip1"]
        im1_i_arr, im1_j_arr, im1_valid_arr = neighbor_cache["im1"]
        jp1_i_arr, jp1_j_arr, jp1_valid_arr = neighbor_cache["jp1"]
        jm1_i_arr, jm1_j_arr, jm1_valid_arr = neighbor_cache["jm1"]
        
        # Get cached effective properties (computed once per tick)
        rho_max_eff = derived.rho_max_eff[i, j]
        eta_eff = derived.eta_eff[i, j]
        opacity_eff = derived.opacity_eff[i, j]
        gravity_strength = cfg.gravity_strength
        shock_k = cfg.shock_k
        rad_absorb_rate = cfg.rad_to_heat_absorb_rate
        eta_damp = eta_eff
        dt_sub = 1.0 / float(M)
        grad_P_x = derived.grad_P_x
        grad_P_y = derived.grad_P_y
        grad_infl_x = derived.grad_infl_x
        grad_infl_y = derived.grad_infl_y
        damp = max(0.0, 1.0 - eta_damp * dt_sub)
        rad_absorb = rad_absorb_rate * dt_sub
        do_shock = shock_k > 0.0

        # skip empty cells
        if rho_array[i, j] <= 1e-12:
            return
        if EH_mask[i, j] > 0.0:
            self.materials.absorb_into_black_hole(self.fabric, i, j, cfg)
            return
        ip1 = int(ip1_i_arr[i, j])
        im1 = int(im1_i_arr[i, j])
        jp1 = int(jp1_j_arr[i, j])
        jm1 = int(jm1_j_arr[i, j])
        ip1_valid = bool(ip1_valid_arr[i, j])
        im1_valid = bool(im1_valid_arr[i, j])
        jp1_valid = bool(jp1_valid_arr[i, j])
        jm1_valid = bool(jm1_valid_arr[i, j])
        dPdx = grad_P_x[i, j]
        dPdy = grad_P_y[i, j]
        dIdx = grad_infl_x[i, j]
        dIdy = grad_infl_y[i, j]
        ledger = self.ledger
        abs_ = abs
        dst_is = [0, 0, 0, 0]
        dst_js = [0, 0, 0, 0]
        dms = [0.0, 0.0, 0.0, 0.0]
        over = max(0.0, rho_array[i, j] - rho_max_eff)
        if (
            abs(dPdx) + abs(dPdy) + abs(dIdx) + abs(dIdy) < 1e-12
            and abs(px[i, j]) + abs(py[i, j]) < 1e-12
            and over <= 0.0
            and not do_shock
            and rad_absorb <= 0.0
        ):
            return
        for _ in range(M):
            rho = rho_array[i, j]
            if rho <= 1e-12:
                break
            rho_safe = rho if rho > 1e-12 else 1e-12
            rho_inv = 1.0 / rho_safe
            # pressure and influence gradients are cached in derived fields
            # accelerations
            a_x = -(dPdx) * rho_inv - gravity_strength * dIdx
            a_y = -(dPdy) * rho_inv - gravity_strength * dIdy
            # update momentum with viscous damping
            px[i, j] += rho * a_x * dt_sub
            py[i, j] += rho * a_y * dt_sub
            px[i, j] *= damp
            py[i, j] *= damp
            # local advection: move mass to neighbours based on velocity
            v_x = px[i, j] * rho_inv
            v_y = py[i, j] * rho_inv
            if (
                abs_(v_x) + abs_(v_y) < 1e-12
                and not do_shock
                and rad_absorb <= 0.0
                and rho <= rho_max_eff
            ):
                break
            # compute outflows to four neighbours
            if v_x > 0.0:
                fx_pos = v_x * dt_sub
                fx_neg = 0.0
            else:
                fx_pos = 0.0
                fx_neg = -v_x * dt_sub
            if v_y > 0.0:
                fy_pos = v_y * dt_sub
                fy_neg = 0.0
            else:
                fy_pos = 0.0
                fy_neg = -v_y * dt_sub
            total_out = fx_pos + fx_neg + fy_pos + fy_neg
            if total_out > 0.5:
                # clamp to avoid numerical explosion
                s = 0.5 / total_out
                fx_pos *= s
                fx_neg *= s
                fy_pos *= s
                fy_neg *= s
            if total_out > 0.0:
                count = 0
                dm = rho * fx_pos
                if dm > 0 and ip1_valid:
                    dst_is[count] = ip1
                    dst_js[count] = j
                    dms[count] = dm
                    count += 1
                dm = rho * fx_neg
                if dm > 0 and im1_valid:
                    dst_is[count] = im1
                    dst_js[count] = j
                    dms[count] = dm
                    count += 1
                dm = rho * fy_pos
                if dm > 0 and jp1_valid:
                    dst_is[count] = i
                    dst_js[count] = jp1
                    dms[count] = dm
                    count += 1
                dm = rho * fy_neg
                if dm > 0 and jm1_valid:
                    dst_is[count] = i
                    dst_js[count] = jm1
                    dms[count] = dm
                    count += 1
                if count:
                    self._apply_mass_transfers_fast(i, j, dst_is, dst_js, dms, count)
            # update local cell after transfers: mass, energies, mixture
            rho = rho_array[i, j]  # updated by transfer
            if rho <= 1e-12:
                break
            # overfill correction: push outwards and heat
            over = rho - rho_max_eff
            if over > 0:
                # push small fraction outward; convert kinetic into heat
                # choose arbitrary direction for repulsion (here just x)
                dm_rep = min(over, rho * 0.1)
                # move to right cell if valid
                if ip1_valid:
                    dst_is[0] = ip1
                    dst_js[0] = j
                    dms[0] = dm_rep
                    self._apply_mass_transfers_fast(i, j, dst_is, dst_js, dms, 1)
                # convert kinetic to heat
                Ekin_before = ledger.kinetic_energy_components(rho, px[i, j], py[i, j])
                # reduce momentum magnitude
                px[i, j] *= 0.9
                py[i, j] *= 0.9
                Ekin_after = ledger.kinetic_energy_components(rho, px[i, j], py[i, j])
                # Compute delta kinetic energy; ensure finite and non-negative.
                if math.isfinite(Ekin_before) and math.isfinite(Ekin_after):
                    raw_dE = Ekin_before - Ekin_after
                    if math.isfinite(raw_dE) and raw_dE > 0.0:
                        dE = raw_dE if raw_dE < ENERGY_DELTA_MAX else ENERGY_DELTA_MAX
                    else:
                        dE = 0.0
                else:
                    dE = 0.0
                # convert to heat; allocate some to radiation based on opacity and T
                if dE > 0.0:
                    T = E_heat[i, j] * rho_inv
                    if T > 0.0 and opacity_eff < 1.0:
                        f_rad = (1.0 - opacity_eff) * (T / (T + 1.0))
                        if f_rad < 0.0:
                            f_rad = 0.0
                        elif f_rad > 1.0:
                            f_rad = 1.0
                    else:
                        f_rad = 0.0
                    E_heat[i, j] += dE * (1.0 - f_rad)
                    E_rad[i, j] += dE * f_rad
            # shock heating due to compression (approximate negative divergence)
            # compute divergence from current velocity field (approx gradient)
            # Use boundary-aware neighbor access
            if do_shock:
                px_ip1 = px[ip1, j] if ip1_valid else px[i, j]
                px_im1 = px[im1, j] if im1_valid else px[i, j]
                py_jp1 = py[i, jp1] if jp1_valid else py[i, j]
                py_jm1 = py[i, jm1] if jm1_valid else py[i, j]

                dvx = (px_ip1 - px_im1) * 0.5
                dvy = (py_jp1 - py_jm1) * 0.5
                div_v = (dvx + dvy) / rho_safe
                if div_v < 0:
                    # compression
                    dE_shock = shock_k * (-div_v) * rho * dt_sub
                    E_heat[i, j] += dE_shock
            # radiation absorption
            if rad_absorb > 0.0:
                absorb = rad_absorb * E_rad[i, j]
                E_rad[i, j] -= absorb
                E_heat[i, j] += absorb

    def _resolve_cells_microtick(
        self,
        active_cells: list[tuple[int, int]],
        derived: DerivedFields,
        tick: int,
    ) -> None:
        """Batch resolve cells for a single microtick."""
        cfg = self.cfg
        rho_array = self.fabric.rho
        px = self.fabric.px
        py = self.fabric.py
        E_heat = self.fabric.E_heat
        E_rad = self.fabric.E_rad
        EH_mask = self.fabric.EH_mask
        neighbor_cache = self.fabric.neighbor_cache
        ip1_i_arr, ip1_j_arr, ip1_valid_arr = neighbor_cache["ip1"]
        im1_i_arr, im1_j_arr, im1_valid_arr = neighbor_cache["im1"]
        jp1_i_arr, jp1_j_arr, jp1_valid_arr = neighbor_cache["jp1"]
        jm1_i_arr, jm1_j_arr, jm1_valid_arr = neighbor_cache["jm1"]

        coords = np.asarray(active_cells, dtype=np.int32)
        i_all = coords[:, 0]
        j_all = coords[:, 1]
        if i_all.size == 0:
            return
        # absorb BH cells up front
        bh_mask = EH_mask[i_all, j_all] > 0.0
        if np.any(bh_mask):
            for idx in np.nonzero(bh_mask)[0]:
                self.materials.absorb_into_black_hole(self.fabric, int(i_all[idx]), int(j_all[idx]), cfg)
        live_mask = ~bh_mask
        if not np.any(live_mask):
            return
        i_idx = i_all[live_mask]
        j_idx = j_all[live_mask]
        rho_vals = rho_array[i_idx, j_idx]
        live_mask = rho_vals > 1e-12
        if not np.any(live_mask):
            return
        i_idx = i_idx[live_mask]
        j_idx = j_idx[live_mask]
        rho_vals = rho_vals[live_mask]

        rho_inv = 1.0 / np.maximum(rho_vals, 1e-12)
        dPdx = derived.grad_P_x[i_idx, j_idx]
        dPdy = derived.grad_P_y[i_idx, j_idx]
        dIdx = derived.grad_infl_x[i_idx, j_idx]
        dIdy = derived.grad_infl_y[i_idx, j_idx]
        a_x = -(dPdx) * rho_inv - cfg.gravity_strength * dIdx
        a_y = -(dPdy) * rho_inv - cfg.gravity_strength * dIdy
        damp = np.maximum(0.0, 1.0 - derived.eta_eff[i_idx, j_idx])
        px[i_idx, j_idx] += rho_vals * a_x
        py[i_idx, j_idx] += rho_vals * a_y
        px[i_idx, j_idx] *= damp
        py[i_idx, j_idx] *= damp

        v_x = px[i_idx, j_idx] * rho_inv
        v_y = py[i_idx, j_idx] * rho_inv
        fx_pos = np.maximum(v_x, 0.0)
        fx_neg = np.maximum(-v_x, 0.0)
        fy_pos = np.maximum(v_y, 0.0)
        fy_neg = np.maximum(-v_y, 0.0)
        total_out = fx_pos + fx_neg + fy_pos + fy_neg
        scale = np.ones_like(total_out)
        np.divide(0.5, total_out, out=scale, where=total_out > 0.5)
        fx_pos *= scale
        fx_neg *= scale
        fy_pos *= scale
        fy_neg *= scale

        dm_ip1 = rho_vals * fx_pos
        dm_im1 = rho_vals * fx_neg
        dm_jp1 = rho_vals * fy_pos
        dm_jm1 = rho_vals * fy_neg
        ip1 = ip1_i_arr[i_idx, j_idx]
        im1 = im1_i_arr[i_idx, j_idx]
        jp1 = jp1_j_arr[i_idx, j_idx]
        jm1 = jm1_j_arr[i_idx, j_idx]
        ip1_valid = ip1_valid_arr[i_idx, j_idx]
        im1_valid = im1_valid_arr[i_idx, j_idx]
        jp1_valid = jp1_valid_arr[i_idx, j_idx]
        jm1_valid = jm1_valid_arr[i_idx, j_idx]

        dst_is = [0, 0, 0, 0]
        dst_js = [0, 0, 0, 0]
        dms = [0.0, 0.0, 0.0, 0.0]
        for idx in range(len(i_idx)):
            count = 0
            if ip1_valid[idx] and dm_ip1[idx] > 0.0:
                dst_is[count] = int(ip1[idx])
                dst_js[count] = int(j_idx[idx])
                dms[count] = float(dm_ip1[idx])
                count += 1
            if im1_valid[idx] and dm_im1[idx] > 0.0:
                dst_is[count] = int(im1[idx])
                dst_js[count] = int(j_idx[idx])
                dms[count] = float(dm_im1[idx])
                count += 1
            if jp1_valid[idx] and dm_jp1[idx] > 0.0:
                dst_is[count] = int(i_idx[idx])
                dst_js[count] = int(jp1[idx])
                dms[count] = float(dm_jp1[idx])
                count += 1
            if jm1_valid[idx] and dm_jm1[idx] > 0.0:
                dst_is[count] = int(i_idx[idx])
                dst_js[count] = int(jm1[idx])
                dms[count] = float(dm_jm1[idx])
                count += 1
            if count:
                self._apply_mass_transfers_fast(int(i_idx[idx]), int(j_idx[idx]), dst_is, dst_js, dms, count)

        shock_k = cfg.shock_k
        rad_absorb_rate = cfg.rad_to_heat_absorb_rate
        do_shock = shock_k > 0.0
        rad_absorb = rad_absorb_rate
        ledger = self.ledger
        for idx in range(len(i_idx)):
            i = int(i_idx[idx])
            j = int(j_idx[idx])
            rho = rho_array[i, j]
            if rho <= 1e-12:
                continue
            rho_inv_cell = 1.0 / max(rho, 1e-12)
            T = E_heat[i, j] * rho_inv_cell
            over = max(0.0, rho - derived.rho_max_eff[i, j])
            if over > 0.0:
                dm_rep = min(over, rho * 0.1)
                if ip1_valid[idx]:
                    dst_is[0] = int(ip1[idx])
                    dst_js[0] = j
                    dms[0] = dm_rep
                    self._apply_mass_transfers_fast(i, j, dst_is, dst_js, dms, 1)
                Ekin_before = ledger.kinetic_energy_components(rho, px[i, j], py[i, j])
                px[i, j] *= 0.9
                py[i, j] *= 0.9
                Ekin_after = ledger.kinetic_energy_components(rho, px[i, j], py[i, j])
                if not math.isfinite(Ekin_before) or not math.isfinite(Ekin_after):
                    dE = 0.0
                else:
                    raw_dE = Ekin_before - Ekin_after
                    if not math.isfinite(raw_dE) or raw_dE <= 0.0:
                        dE = 0.0
                    else:
                        dE = min(raw_dE, ENERGY_DELTA_MAX)
                if T > 0.0 and derived.opacity_eff[i, j] < 1.0:
                    f_rad = (1.0 - derived.opacity_eff[i, j]) * (T / (T + 1.0))
                    if f_rad < 0.0:
                        f_rad = 0.0
                    elif f_rad > 1.0:
                        f_rad = 1.0
                else:
                    f_rad = 0.0
                ledger.convert_kinetic_to_heat(i, j, dE, f_rad)
            if do_shock:
                px_ip1 = px[int(ip1[idx]), j] if ip1_valid[idx] else px[i, j]
                px_im1 = px[int(im1[idx]), j] if im1_valid[idx] else px[i, j]
                py_jp1 = py[i, int(jp1[idx])] if jp1_valid[idx] else py[i, j]
                py_jm1 = py[i, int(jm1[idx])] if jm1_valid[idx] else py[i, j]
                dvx = (px_ip1 - px_im1) * 0.5
                dvy = (py_jp1 - py_jm1) * 0.5
                div_v = (dvx + dvy) / max(rho, 1e-12)
                if div_v < 0:
                    E_heat[i, j] += shock_k * (-div_v) * rho
            if rad_absorb > 0.0:
                absorb = rad_absorb * E_rad[i, j]
                E_rad[i, j] -= absorb
                E_heat[i, j] += absorb

    def transfer_mass(self, src_i: int, src_j: int, dst_i: int, dst_j: int, mass: float, v_x: float, v_y: float) -> None:
        """Move a quantity of mass from ``src`` to ``dst`` with associated momentum and energy.

        ``mass`` is removed from the source cell's density and added to
        the destination. Momentum is scaled accordingly using the
        current velocity at the source. Heat and radiation are moved
        proportionally. Species masses are moved proportionally as well.
        Mixtures are maintained with top-K filtering.
        """
        if mass <= 0.0:
            return
        # clamp mass to available
        rho = self.fabric.rho
        px = self.fabric.px
        py = self.fabric.py
        E_heat = self.fabric.E_heat
        E_rad = self.fabric.E_rad
        mixtures = self.fabric.mixtures
        avail = rho[src_i, src_j]
        if avail <= 0.0:
            return
        if mass > avail:
            mass = avail
        # remove from source
        rho[src_i, src_j] -= mass
        # compute fraction moved
        frac = mass / avail if avail > 0 else 0
        # move momentum
        dp_x = px[src_i, src_j] * frac
        dp_y = py[src_i, src_j] * frac
        px[src_i, src_j] -= dp_x
        py[src_i, src_j] -= dp_y
        # add to destination
        rho[dst_i, dst_j] += mass
        px[dst_i, dst_j] += dp_x
        py[dst_i, dst_j] += dp_y
        # move heat and radiation
        dE_heat = E_heat[src_i, src_j] * frac
        dE_rad = E_rad[src_i, src_j] * frac
        E_heat[src_i, src_j] -= dE_heat
        E_rad[src_i, src_j] -= dE_rad
        E_heat[dst_i, dst_j] += dE_heat
        E_rad[dst_i, dst_j] += dE_rad
        # move mixture
        src_mix = mixtures[src_i][src_j]
        dst_mix = mixtures[dst_i][dst_j]
        mix_total = src_mix.total_mass()
        if mix_total > 0.0:
            top_k = self.cfg.mixture_top_k
            frac_mix = frac
            if getattr(src_mix, "_array_mode", False):
                for idx in range(src_mix._count):
                    sid = src_mix._species_ids_arr[idx]
                    mval = src_mix._masses_arr[idx]
                    moved = mval * frac_mix
                    src_mix._masses_arr[idx] -= moved
                    dst_mix.add_species_mass(sid, moved, top_k)
                src_mix._total_mass -= mix_total * frac_mix
            else:
                for idx in range(len(src_mix.species_ids)):
                    sid = src_mix.species_ids[idx]
                    mval = src_mix.masses[idx]
                    moved = mval * frac_mix
                    src_mix.masses[idx] -= moved
                    dst_mix.add_species_mass(sid, moved, top_k)
            self.fabric.mark_mixture_dirty(src_i, src_j)
            self.fabric.mark_mixture_dirty(dst_i, dst_j)

    def _apply_mass_transfers_fast(
        self,
        src_i: int,
        src_j: int,
        dst_is: list[int],
        dst_js: list[int],
        dms: list[float],
        count: int,
    ) -> None:
        """Batch apply multiple transfers from a single source cell."""
        rho = self.fabric.rho
        px = self.fabric.px
        py = self.fabric.py
        E_heat = self.fabric.E_heat
        E_rad = self.fabric.E_rad
        mixtures = self.fabric.mixtures

        if count <= 0:
            return
        avail = rho[src_i, src_j]
        if avail <= 0.0:
            return
        if count == 1:
            dm = dms[0]
            if dm <= 0.0:
                return
            if dm > avail:
                dm = avail
            frac = dm / avail
            dst_i = dst_is[0]
            dst_j = dst_js[0]
            src_px = px[src_i, src_j]
            src_py = py[src_i, src_j]
            src_E_heat = E_heat[src_i, src_j]
            src_E_rad = E_rad[src_i, src_j]
            rho[dst_i, dst_j] += dm
            px[dst_i, dst_j] += src_px * frac
            py[dst_i, dst_j] += src_py * frac
            E_heat[dst_i, dst_j] += src_E_heat * frac
            E_rad[dst_i, dst_j] += src_E_rad * frac
            rho[src_i, src_j] -= dm
            px[src_i, src_j] = src_px * (1.0 - frac)
            py[src_i, src_j] = src_py * (1.0 - frac)
            E_heat[src_i, src_j] = src_E_heat * (1.0 - frac)
            E_rad[src_i, src_j] = src_E_rad * (1.0 - frac)
            src_mix = mixtures[src_i][src_j]
            mix_total = src_mix.total_mass()
            if mix_total > 0.0:
                top_k = self.cfg.mixture_top_k
                dst_mix = mixtures[dst_i][dst_j]
                if getattr(src_mix, "_array_mode", False):
                    for idx in range(src_mix._count):
                        sid = src_mix._species_ids_arr[idx]
                        mval = src_mix._masses_arr[idx]
                        if mval <= 0:
                            continue
                        moved = mval * frac
                        src_mix._masses_arr[idx] -= moved
                        dst_mix.add_species_mass(sid, moved, top_k)
                    src_mix._total_mass -= mix_total * frac
                else:
                    for idx in range(len(src_mix.species_ids)):
                        sid = src_mix.species_ids[idx]
                        mval = src_mix.masses[idx]
                        if mval <= 0:
                            continue
                        moved = mval * frac
                        src_mix.masses[idx] -= moved
                        dst_mix.add_species_mass(sid, moved, top_k)
                self.fabric.mark_mixture_dirty(src_i, src_j)
                self.fabric.mark_mixture_dirty(dst_i, dst_j)
            return
        total_dm = 0.0
        for idx in range(count):
            total_dm += dms[idx]
        if total_dm <= 0.0:
            return
        if total_dm > avail:
            scale = avail / total_dm
            for idx in range(count):
                dms[idx] *= scale
            total_dm = avail

        inv_avail = 1.0 / avail
        frac_total = total_dm * inv_avail
        src_px = px[src_i, src_j]
        src_py = py[src_i, src_j]
        src_E_heat = E_heat[src_i, src_j]
        src_E_rad = E_rad[src_i, src_j]

        frac_list = [0.0] * count
        for idx in range(count):
            dm = dms[idx]
            frac = dm * inv_avail
            frac_list[idx] = frac
            dst_i = dst_is[idx]
            dst_j = dst_js[idx]
            rho[dst_i, dst_j] += dm
            px[dst_i, dst_j] += src_px * frac
            py[dst_i, dst_j] += src_py * frac
            E_heat[dst_i, dst_j] += src_E_heat * frac
            E_rad[dst_i, dst_j] += src_E_rad * frac

        rho[src_i, src_j] -= total_dm
        px[src_i, src_j] = src_px * (1.0 - frac_total)
        py[src_i, src_j] = src_py * (1.0 - frac_total)
        E_heat[src_i, src_j] = src_E_heat * (1.0 - frac_total)
        E_rad[src_i, src_j] = src_E_rad * (1.0 - frac_total)

        src_mix = mixtures[src_i][src_j]
        mix_total = src_mix.total_mass()
        if mix_total > 0.0:
            top_k = self.cfg.mixture_top_k
            dst_mixes = [mixtures[dst_is[idx]][dst_js[idx]] for idx in range(count)]
            frac_total_mix = frac_total
            if getattr(src_mix, "_array_mode", False):
                for idx in range(src_mix._count):
                    sid = src_mix._species_ids_arr[idx]
                    mval = src_mix._masses_arr[idx]
                    if mval <= 0:
                        continue
                    for mix_idx in range(count):
                        dst_mixes[mix_idx].add_species_mass(sid, mval * frac_list[mix_idx], top_k)
                src_mix._masses_arr[:src_mix._count] *= (1.0 - frac_total_mix)
                src_mix._total_mass *= (1.0 - frac_total_mix)
            else:
                for idx in range(len(src_mix.species_ids)):
                    sid = src_mix.species_ids[idx]
                    mval = src_mix.masses[idx]
                    if mval <= 0:
                        continue
                    for mix_idx in range(count):
                        dst_mixes[mix_idx].add_species_mass(sid, mval * frac_list[mix_idx], top_k)
                for idx in range(len(src_mix.masses)):
                    src_mix.masses[idx] *= (1.0 - frac_total_mix)
            self.fabric.mark_mixture_dirty(src_i, src_j)
            for idx in range(count):
                self.fabric.mark_mixture_dirty(dst_is[idx], dst_js[idx])

    def _apply_mass_transfers(self, src_i: int, src_j: int, transfers: list[tuple[int, int, float]]) -> None:
        """Batch apply multiple transfers from a single source cell."""
        if not transfers:
            return
        count = len(transfers)
        dst_is = [0] * count
        dst_js = [0] * count
        dms = [0.0] * count
        for idx, (di, dj, dm) in enumerate(transfers):
            dst_is[idx] = di
            dst_js[idx] = dj
            dms[idx] = dm
        self._apply_mass_transfers_fast(src_i, src_j, dst_is, dst_js, dms, count)
