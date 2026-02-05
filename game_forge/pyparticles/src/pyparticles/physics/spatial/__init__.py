"""
Eidosian PyParticles V6 - Spatial Data Structures

High-performance spatial indexing for O(N) neighbor queries:
- Uniform Grid (current implementation)
- Morton/Z-order encoding for cache efficiency
- Adaptive cell sizing
"""

import numpy as np
from numba import njit, prange
from dataclasses import dataclass
from typing import Tuple


@dataclass
class GridConfig:
    """Configuration for uniform spatial grid."""
    cell_size: float
    grid_width: int
    grid_height: int
    max_per_cell: int
    half_world: float
    
    @classmethod
    def from_world(cls, world_size: float, interaction_radius: float, 
                   n_particles: int, density_factor: float = 20.0):
        """
        Create grid config from world parameters.
        
        Args:
            world_size: Total world dimension
            interaction_radius: Max interaction radius (determines cell size)
            n_particles: Number of particles (for capacity estimation)
            density_factor: Safety multiplier for max_per_cell
        """
        half_world = world_size / 2.0
        cell_size = max(interaction_radius, 0.05 * world_size)
        
        grid_width = int(world_size / cell_size) + 2
        grid_height = int(world_size / cell_size) + 2
        
        avg_density = n_particles / (grid_width * grid_height)
        max_per_cell = int(avg_density * density_factor) + 100
        
        return cls(
            cell_size=cell_size,
            grid_width=grid_width,
            grid_height=grid_height,
            max_per_cell=max_per_cell,
            half_world=half_world
        )


@njit(fastmath=True, cache=True, inline='always')
def morton_encode(x: int, y: int) -> int:
    """
    Encode 2D coordinates into Morton/Z-order code.
    
    Morton ordering improves cache locality for spatial queries
    by keeping nearby cells close in memory.
    """
    x_bits = x & 0xFFFF
    y_bits = y & 0xFFFF
    
    x_bits = (x_bits | (x_bits << 8)) & 0x00FF00FF
    x_bits = (x_bits | (x_bits << 4)) & 0x0F0F0F0F
    x_bits = (x_bits | (x_bits << 2)) & 0x33333333
    x_bits = (x_bits | (x_bits << 1)) & 0x55555555
    
    y_bits = (y_bits | (y_bits << 8)) & 0x00FF00FF
    y_bits = (y_bits | (y_bits << 4)) & 0x0F0F0F0F
    y_bits = (y_bits | (y_bits << 2)) & 0x33333333
    y_bits = (y_bits | (y_bits << 1)) & 0x55555555
    
    return x_bits | (y_bits << 1)


@njit(fastmath=True, cache=True, inline='always')
def morton_decode(z: int) -> Tuple[int, int]:
    """Decode Morton code back to x, y coordinates."""
    x = z & 0x55555555
    y = (z >> 1) & 0x55555555
    
    x = (x | (x >> 1)) & 0x33333333
    x = (x | (x >> 2)) & 0x0F0F0F0F
    x = (x | (x >> 4)) & 0x00FF00FF
    x = (x | (x >> 8)) & 0x0000FFFF
    
    y = (y | (y >> 1)) & 0x33333333
    y = (y | (y >> 2)) & 0x0F0F0F0F
    y = (y | (y >> 4)) & 0x00FF00FF
    y = (y | (y >> 8)) & 0x0000FFFF
    
    return x, y


@njit(parallel=True, fastmath=True, cache=True)
def compute_morton_order(pos: np.ndarray, n_active: int, 
                         cell_size: float, half_world: float,
                         grid_w: int, grid_h: int) -> np.ndarray:
    """
    Compute Morton ordering for particles.
    
    Returns array of indices that would sort particles in Morton order.
    This improves cache efficiency during force computation.
    """
    morton_codes = np.empty(n_active, dtype=np.uint32)
    
    for i in prange(n_active):
        cx = int((pos[i, 0] + half_world) / cell_size)
        cy = int((pos[i, 1] + half_world) / cell_size)
        
        cx = max(0, min(cx, grid_w - 1))
        cy = max(0, min(cy, grid_h - 1))
        
        morton_codes[i] = morton_encode(cx, cy)
    
    return np.argsort(morton_codes)


@njit(fastmath=True, cache=True)
def fill_grid_morton(pos: np.ndarray, morton_order: np.ndarray,
                     n_active: int, cell_size: float,
                     grid_counts: np.ndarray, grid_cells: np.ndarray,
                     half_world: float):
    """
    Fill spatial grid using Morton-ordered particles.
    
    Processing particles in Morton order improves cache efficiency.
    """
    grid_counts.fill(0)
    h, w = grid_counts.shape
    max_p = grid_cells.shape[2]
    
    for k in range(n_active):
        i = morton_order[k]
        
        cx = int((pos[i, 0] + half_world) / cell_size)
        cy = int((pos[i, 1] + half_world) / cell_size)
        
        if cx < 0: cx = 0
        elif cx >= w: cx = w - 1
        if cy < 0: cy = 0
        elif cy >= h: cy = h - 1
        
        idx = grid_counts[cy, cx]
        if idx < max_p:
            grid_cells[cy, cx, idx] = i
            grid_counts[cy, cx] += 1


@njit(fastmath=True, cache=True)
def compute_cell_densities(grid_counts: np.ndarray) -> Tuple[float, float, int]:
    """
    Compute grid density statistics.
    
    Returns: (avg_density, max_density, num_empty_cells)
    """
    h, w = grid_counts.shape
    total = 0
    max_count = 0
    empty_count = 0
    
    for y in range(h):
        for x in range(w):
            c = grid_counts[y, x]
            total += c
            if c > max_count:
                max_count = c
            if c == 0:
                empty_count += 1
    
    n_cells = h * w
    avg = float(total) / float(n_cells) if n_cells > 0 else 0.0
    
    return avg, float(max_count), empty_count


@njit(fastmath=True, cache=True)
def adaptive_cell_size(avg_density: float, max_density: float,
                        current_cell_size: float, target_avg: float = 10.0,
                        min_cell_size: float = 0.1, max_cell_size: float = 5.0) -> float:
    """
    Adaptively adjust cell size based on density.
    
    If cells are too dense, increase cell size.
    If cells are too sparse, decrease cell size.
    """
    if max_density > target_avg * 5:
        return min(current_cell_size * 1.2, max_cell_size)
    elif avg_density < target_avg * 0.2:
        return max(current_cell_size * 0.8, min_cell_size)
    
    return current_cell_size


# =============================================================================
# SIMD-FRIENDLY DATA LAYOUT (Structure of Arrays)
# =============================================================================

@njit(fastmath=True, cache=True)
def pack_positions_soa(pos: np.ndarray, n_active: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pack positions into Structure-of-Arrays format for SIMD.
    
    SoA layout: x_arr[N], y_arr[N]
    Better for vectorization than AoS: pos[N, 2]
    """
    x_arr = np.empty(n_active, dtype=np.float32)
    y_arr = np.empty(n_active, dtype=np.float32)
    
    for i in range(n_active):
        x_arr[i] = pos[i, 0]
        y_arr[i] = pos[i, 1]
    
    return x_arr, y_arr


@njit(fastmath=True, cache=True)
def unpack_positions_aos(x_arr: np.ndarray, y_arr: np.ndarray, 
                          pos: np.ndarray, n_active: int):
    """
    Unpack positions from SoA back to AoS format.
    """
    for i in range(n_active):
        pos[i, 0] = x_arr[i]
        pos[i, 1] = y_arr[i]


# =============================================================================
# BATCH PROCESSING UTILITIES
# =============================================================================

@njit(fastmath=True, cache=True)
def compute_batch_ranges(n_active: int, batch_size: int) -> np.ndarray:
    """
    Compute start/end indices for batch processing.
    
    Returns array of shape (num_batches, 2) with [start, end] for each batch.
    """
    num_batches = (n_active + batch_size - 1) // batch_size
    ranges = np.empty((num_batches, 2), dtype=np.int32)
    
    for b in range(num_batches):
        start = b * batch_size
        end = min(start + batch_size, n_active)
        ranges[b, 0] = start
        ranges[b, 1] = end
    
    return ranges


@njit(parallel=True, fastmath=True, cache=True)
def prefetch_neighbor_data(pos: np.ndarray, colors: np.ndarray, angle: np.ndarray,
                            cell_particles: np.ndarray, cell_count: int,
                            max_neighbors: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prefetch neighbor data into contiguous arrays for cache efficiency.
    
    Returns (positions, colors, angles) for up to max_neighbors particles in cell.
    """
    actual_count = min(cell_count, max_neighbors)
    
    n_pos = np.empty((actual_count, 2), dtype=np.float32)
    n_colors = np.empty(actual_count, dtype=np.int32)
    n_angles = np.empty(actual_count, dtype=np.float32)
    
    for k in prange(actual_count):
        idx = cell_particles[k]
        n_pos[k, 0] = pos[idx, 0]
        n_pos[k, 1] = pos[idx, 1]
        n_colors[k] = colors[idx]
        n_angles[k] = angle[idx]
    
    return n_pos, n_colors, n_angles
