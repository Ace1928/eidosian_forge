"""
Canonical ordering module for Stratum.

This module defines the iteration ordering rules that all determinism
guarantees rest upon. Every component that iterates over cells, patches,
events, or signals must use these ordering functions.

Ordering Guarantees:
- STRICT mode: Lexicographic ordering (i, j) for cells
- Patches: Sorted by (chunk_i, chunk_j) 
- Events within tick: Sorted by (type, cell, timestamp)
- RNG streams: Partitioned by (tick, patch_id, event_id)

Usage:
    from core.ordering import get_cell_order, get_patch_order
    
    for i, j in get_cell_order(active_cells, mode):
        process_cell(i, j)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Iterator, Callable, Any
from enum import Enum

import numpy as np

from .config import DeterminismMode


def lexicographic_cell_key(cell: Tuple[int, int]) -> Tuple[int, int]:
    """Key function for lexicographic cell ordering.
    
    Cells are sorted first by i (row), then by j (column).
    This provides a stable, deterministic ordering.
    """
    return (cell[0], cell[1])


def score_based_cell_key(
    cell: Tuple[int, int],
    scores: np.ndarray
) -> Tuple[float, int, int]:
    """Key function for score-based cell ordering with lexicographic tiebreaker.
    
    Cells are sorted by descending score, with lexicographic ordering as tiebreaker.
    This ensures deterministic ordering even when scores are equal.
    """
    i, j = cell
    return (-scores[i, j], i, j)


def get_cell_order(
    cells: List[Tuple[int, int]],
    mode: DeterminismMode,
    scores: np.ndarray = None
) -> List[Tuple[int, int]]:
    """Get deterministically ordered list of cells.
    
    Args:
        cells: List of (i, j) cell coordinates.
        mode: Determinism mode controlling ordering strategy.
        scores: Optional score array for priority-based ordering.
        
    Returns:
        Ordered list of cells according to the specified mode.
    """
    if mode == DeterminismMode.STRICT_DETERMINISTIC:
        # Always use lexicographic ordering in strict mode
        return sorted(cells, key=lexicographic_cell_key)
    elif mode == DeterminismMode.REPLAY_DETERMINISTIC:
        # Use score-based ordering with lexicographic tiebreaker
        if scores is not None:
            return sorted(cells, key=lambda c: score_based_cell_key(c, scores))
        return sorted(cells, key=lexicographic_cell_key)
    else:
        # REALTIME_ADAPTIVE: score-based without strict ordering guarantee
        if scores is not None:
            return sorted(cells, key=lambda c: (-scores[c[0], c[1]], c[0], c[1]))
        return list(cells)


@dataclass(frozen=True)
class Patch:
    """A rectangular region of the grid.
    
    Patches are used for locality-aware scheduling and potential
    parallel execution. Each patch has a unique ID based on its
    position in the patch grid.
    """
    patch_i: int  # Patch row index
    patch_j: int  # Patch column index
    start_i: int  # Grid start row
    start_j: int  # Grid start column
    end_i: int    # Grid end row (exclusive)
    end_j: int    # Grid end column (exclusive)
    
    @property
    def patch_id(self) -> Tuple[int, int]:
        """Unique patch identifier."""
        return (self.patch_i, self.patch_j)
    
    def cells(self) -> Iterator[Tuple[int, int]]:
        """Iterate over cells in lexicographic order."""
        for i in range(self.start_i, self.end_i):
            for j in range(self.start_j, self.end_j):
                yield (i, j)
    
    def __lt__(self, other: 'Patch') -> bool:
        """Comparison for sorting patches."""
        return self.patch_id < other.patch_id


def create_patches(
    grid_w: int,
    grid_h: int,
    patch_size: int = 16
) -> List[Patch]:
    """Create a list of patches covering the grid.
    
    Args:
        grid_w: Grid width.
        grid_h: Grid height.
        patch_size: Size of each patch (square).
        
    Returns:
        List of Patch objects in lexicographic order.
    """
    patches = []
    
    n_patches_i = (grid_w + patch_size - 1) // patch_size
    n_patches_j = (grid_h + patch_size - 1) // patch_size
    
    for pi in range(n_patches_i):
        for pj in range(n_patches_j):
            start_i = pi * patch_size
            start_j = pj * patch_size
            end_i = min(start_i + patch_size, grid_w)
            end_j = min(start_j + patch_size, grid_h)
            
            patches.append(Patch(
                patch_i=pi,
                patch_j=pj,
                start_i=start_i,
                start_j=start_j,
                end_i=end_i,
                end_j=end_j
            ))
    
    return sorted(patches)


def get_patch_order(
    patches: List[Patch],
    mode: DeterminismMode,
    activity_scores: np.ndarray = None
) -> List[Patch]:
    """Get deterministically ordered list of patches.
    
    Args:
        patches: List of patches to order.
        mode: Determinism mode.
        activity_scores: Optional per-patch activity scores.
        
    Returns:
        Ordered list of patches.
    """
    if mode == DeterminismMode.STRICT_DETERMINISTIC:
        # Always lexicographic in strict mode
        return sorted(patches)
    elif mode == DeterminismMode.REPLAY_DETERMINISTIC:
        # Can use activity-based ordering with stable tiebreaker
        if activity_scores is not None:
            # Sort by descending activity, then by patch_id
            def key(p):
                score = np.sum(activity_scores[p.start_i:p.end_i, p.start_j:p.end_j])
                return (-score, p.patch_i, p.patch_j)
            return sorted(patches, key=key)
        return sorted(patches)
    else:
        # REALTIME_ADAPTIVE: activity-based
        if activity_scores is not None:
            def key(p):
                return -np.sum(activity_scores[p.start_i:p.end_i, p.start_j:p.end_j])
            return sorted(patches, key=key)
        return list(patches)


class RNGStreamPartitioner:
    """Partitions RNG streams for deterministic parallel execution.
    
    Each stream is derived from a base seed plus identifiers for:
    - tick
    - patch_id
    - event_id (within patch)
    
    This ensures that parallel execution produces deterministic results
    as long as the same partitioning is used.
    """
    
    def __init__(self, base_seed: int):
        """Initialize the partitioner.
        
        Args:
            base_seed: Base seed for all streams.
        """
        self.base_seed = base_seed
    
    def get_stream_seed(
        self,
        tick: int,
        patch_id: Tuple[int, int] = (0, 0),
        event_id: int = 0
    ) -> int:
        """Get a deterministic seed for a specific stream.
        
        The seed is computed using a stable hash of all identifiers.
        
        Args:
            tick: Simulation tick.
            patch_id: Patch identifier (pi, pj).
            event_id: Event index within the patch.
            
        Returns:
            Deterministic seed for numpy.random.default_rng().
        """
        import hashlib
        import struct
        
        h = hashlib.blake2s(digest_size=8)
        h.update(struct.pack('>q', self.base_seed))
        h.update(struct.pack('>i', tick))
        h.update(struct.pack('>ii', patch_id[0], patch_id[1]))
        h.update(struct.pack('>i', event_id))
        
        return int.from_bytes(h.digest(), byteorder='big', signed=False)
    
    def get_rng(
        self,
        tick: int,
        patch_id: Tuple[int, int] = (0, 0),
        event_id: int = 0
    ) -> np.random.Generator:
        """Get a numpy random generator for a specific stream.
        
        Args:
            tick: Simulation tick.
            patch_id: Patch identifier.
            event_id: Event index.
            
        Returns:
            numpy.random.Generator initialized with deterministic seed.
        """
        seed = self.get_stream_seed(tick, patch_id, event_id)
        return np.random.default_rng(seed)


class EventOrdering:
    """Ordering rules for events within a tick.
    
    Events are sorted by:
    1. Event type priority (some events must happen before others)
    2. Cell location (lexicographic)
    3. Timestamp (if available)
    """
    
    # Event type priorities (lower = earlier)
    TYPE_PRIORITIES = {
        'signal_delivery': 0,
        'force_computation': 1,
        'advection': 2,
        'diffusion': 3,
        'high_energy': 4,
        'boundary_flux': 5,
    }
    
    @classmethod
    def event_key(
        cls,
        event_type: str,
        cell: Tuple[int, int],
        timestamp: float = 0.0
    ) -> Tuple[int, int, int, float]:
        """Generate a sort key for an event.
        
        Args:
            event_type: Type of event.
            cell: Cell location.
            timestamp: Event timestamp.
            
        Returns:
            Tuple suitable for sorting.
        """
        priority = cls.TYPE_PRIORITIES.get(event_type, 100)
        return (priority, cell[0], cell[1], timestamp)
    
    @classmethod
    def sort_events(
        cls,
        events: List[Tuple[str, Tuple[int, int], Any]],
        mode: DeterminismMode
    ) -> List[Tuple[str, Tuple[int, int], Any]]:
        """Sort events according to determinism mode.
        
        Args:
            events: List of (event_type, cell, data) tuples.
            mode: Determinism mode.
            
        Returns:
            Sorted list of events.
        """
        if mode == DeterminismMode.STRICT_DETERMINISTIC:
            return sorted(events, key=lambda e: cls.event_key(e[0], e[1]))
        return list(events)


# Module-level utilities for common operations

def iterate_grid_lexicographic(
    grid_w: int,
    grid_h: int
) -> Iterator[Tuple[int, int]]:
    """Iterate over all grid cells in lexicographic order.
    
    Args:
        grid_w: Grid width.
        grid_h: Grid height.
        
    Yields:
        (i, j) cell coordinates.
    """
    for i in range(grid_w):
        for j in range(grid_h):
            yield (i, j)


def select_top_k_cells_deterministic(
    scores: np.ndarray,
    k: int,
    mode: DeterminismMode
) -> List[Tuple[int, int]]:
    """Select top K cells by score with deterministic ordering.
    
    Uses argpartition for O(N) selection, then sorts the top K
    with lexicographic tiebreaker for determinism.
    
    Args:
        scores: 2D array of scores.
        k: Number of cells to select.
        mode: Determinism mode.
        
    Returns:
        List of (i, j) cell coordinates, highest scores first.
    """
    W, H = scores.shape
    flat_scores = scores.ravel()
    n_total = len(flat_scores)
    k = min(k, n_total)
    
    if k <= 0:
        return []
    
    if k >= n_total:
        # Return all cells
        all_cells = [(i, j) for i in range(W) for j in range(H)]
        return get_cell_order(all_cells, mode, scores)
    
    # Use argpartition for O(N) top-K selection
    partition_idx = n_total - k
    partitioned = np.argpartition(flat_scores, partition_idx)
    top_k_flat = partitioned[partition_idx:]
    
    # Convert to (i, j) coordinates
    cells = [(int(idx // H), int(idx % H)) for idx in top_k_flat]
    
    # Sort according to mode
    return get_cell_order(cells, mode, scores)
