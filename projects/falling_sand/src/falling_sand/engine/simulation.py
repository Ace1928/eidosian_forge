"""Simulation step logic for the falling sand engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from falling_sand.engine.chunk import Chunk
from falling_sand.engine.materials import Material, MOVES_DOWN, MOVES_UP


ChunkCoord = Tuple[int, int, int]


@dataclass(frozen=True)
class StepStats:
    """Statistics for a simulation step."""

    moved_voxels: int
    touched_chunks: int


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for simulation behavior."""

    granular_spread_chance: float = 0.35
    liquid_spread_chance: float = 0.7
    gas_spread_chance: float = 0.85
    rng_seed: int | None = None

    def __post_init__(self) -> None:
        _validate_chance(self.granular_spread_chance, "granular_spread_chance")
        _validate_chance(self.liquid_spread_chance, "liquid_spread_chance")
        _validate_chance(self.gas_spread_chance, "gas_spread_chance")


def _validate_chance(value: float, field: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1")


def _move_down(data: np.ndarray, materials: Iterable[Material]) -> tuple[np.ndarray, int]:
    size = data.shape[2]
    if size <= 1:
        return data.copy(), 0

    result = data.copy()
    moved = 0
    mask = np.isin(data, [int(mat) for mat in materials])
    can_move = mask[:, :, 1:] & (data[:, :, :-1] == int(Material.AIR))
    moved = int(np.count_nonzero(can_move))
    result[:, :, :-1][can_move] = data[:, :, 1:][can_move]
    result[:, :, 1:][can_move] = int(Material.AIR)
    return result, moved


def _move_up(data: np.ndarray, materials: Iterable[Material]) -> tuple[np.ndarray, int]:
    size = data.shape[2]
    if size <= 1:
        return data.copy(), 0

    result = data.copy()
    mask = np.isin(data, [int(mat) for mat in materials])
    can_move = mask[:, :, :-1] & (data[:, :, 1:] == int(Material.AIR))
    moved = int(np.count_nonzero(can_move))
    result[:, :, 1:][can_move] = data[:, :, :-1][can_move]
    result[:, :, :-1][can_move] = int(Material.AIR)
    return result, moved


def _move_lateral(
    data: np.ndarray,
    materials: Iterable[Material],
    axis: int,
    rng: np.random.Generator,
    chance: float,
) -> tuple[np.ndarray, int]:
    if chance <= 0.0:
        return data.copy(), 0
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")

    result = data.copy()
    mat_ids = [int(mat) for mat in materials]
    moved = 0

    def shift_once(direction: int) -> int:
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1")
        src_slice = [slice(None)] * 3
        dst_slice = [slice(None)] * 3
        if direction == 1:
            src_slice[axis] = slice(None, -1)
            dst_slice[axis] = slice(1, None)
        else:
            src_slice[axis] = slice(1, None)
            dst_slice[axis] = slice(None, -1)

        src = result[tuple(src_slice)]
        dst = result[tuple(dst_slice)]
        src_mask = np.isin(src, mat_ids)
        dst_mask = dst == int(Material.AIR)
        random_mask = rng.random(src.shape) < chance
        move_mask = src_mask & dst_mask & random_mask
        count = int(np.count_nonzero(move_mask))
        if count:
            dst[move_mask] = src[move_mask]
            src[move_mask] = int(Material.AIR)
        return count

    moved += shift_once(1)
    moved += shift_once(-1)
    return result, moved


def _move_down_diagonal(
    data: np.ndarray,
    rng: np.random.Generator,
    chance: float,
) -> tuple[np.ndarray, int]:
    if chance <= 0.0:
        return data.copy(), 0
    size = data.shape[2]
    if size <= 1:
        return data.copy(), 0

    result = data.copy()
    moved = 0

    def attempt(axis: int, direction: int) -> int:
        src_slice = [slice(None)] * 3
        dst_slice = [slice(None)] * 3
        below_slice = [slice(None)] * 3
        if axis == 0:
            if direction == -1:
                src_slice[0] = slice(1, None)
                dst_slice[0] = slice(None, -1)
                below_slice[0] = slice(1, None)
            else:
                src_slice[0] = slice(None, -1)
                dst_slice[0] = slice(1, None)
                below_slice[0] = slice(None, -1)
        else:
            if direction == -1:
                src_slice[1] = slice(1, None)
                dst_slice[1] = slice(None, -1)
                below_slice[1] = slice(1, None)
            else:
                src_slice[1] = slice(None, -1)
                dst_slice[1] = slice(1, None)
                below_slice[1] = slice(None, -1)

        src_slice[2] = slice(1, None)
        dst_slice[2] = slice(None, -1)
        below_slice[2] = slice(None, -1)

        src = result[tuple(src_slice)]
        dst = result[tuple(dst_slice)]
        below = result[tuple(below_slice)]
        src_mask = src == int(Material.GRANULAR)
        blocked = below != int(Material.AIR)
        dst_mask = dst == int(Material.AIR)
        random_mask = rng.random(src.shape) < chance
        move_mask = src_mask & blocked & dst_mask & random_mask
        count = int(np.count_nonzero(move_mask))
        if count:
            dst[move_mask] = src[move_mask]
            src[move_mask] = int(Material.AIR)
        return count

    moved += attempt(0, -1)
    moved += attempt(0, 1)
    moved += attempt(1, -1)
    moved += attempt(1, 1)
    return result, moved


def step_chunk_vertical(chunk: Chunk, rng: np.random.Generator, config: SimulationConfig) -> tuple[np.ndarray, int]:
    """Apply vertical movement for granular/liquid/gas within a chunk."""

    data, moved_down = _move_down(chunk.data, MOVES_DOWN)
    data, moved_up = _move_up(data, MOVES_UP)
    data, moved_diag = _move_down_diagonal(data, rng, config.granular_spread_chance)
    return data, moved_down + moved_up + moved_diag


def step_chunk_lateral(
    chunk: Chunk,
    rng: np.random.Generator,
    config: SimulationConfig,
) -> tuple[np.ndarray, int]:
    """Apply lateral movement for liquids and gases within a chunk."""

    data, moved_liq_x = _move_lateral(
        chunk.data, (Material.LIQUID, Material.GRANULAR), axis=0, rng=rng, chance=config.liquid_spread_chance
    )
    data, moved_liq_y = _move_lateral(
        data, (Material.LIQUID, Material.GRANULAR), axis=1, rng=rng, chance=config.liquid_spread_chance
    )
    data, moved_gas_x = _move_lateral(
        data, (Material.GAS,), axis=0, rng=rng, chance=config.gas_spread_chance
    )
    data, moved_gas_y = _move_lateral(
        data, (Material.GAS,), axis=1, rng=rng, chance=config.gas_spread_chance
    )
    return data, moved_liq_x + moved_liq_y + moved_gas_x + moved_gas_y


def apply_boundary_transfers(
    chunks: Dict[ChunkCoord, Chunk],
    coord: ChunkCoord,
) -> int:
    """Move voxels across chunk boundaries."""

    chunk = chunks[coord]
    size = chunk.data.shape[2]
    moved = 0
    below = chunks.get((coord[0], coord[1], coord[2] - 1))
    above = chunks.get((coord[0], coord[1], coord[2] + 1))
    left = chunks.get((coord[0] - 1, coord[1], coord[2]))
    right = chunks.get((coord[0] + 1, coord[1], coord[2]))
    back = chunks.get((coord[0], coord[1] - 1, coord[2]))
    front = chunks.get((coord[0], coord[1] + 1, coord[2]))

    if below is not None:
        bottom = chunk.data[:, :, 0]
        below_top = below.data[:, :, size - 1]
        movable = np.isin(bottom, [int(mat) for mat in MOVES_DOWN])
        can_move = movable & (below_top == int(Material.AIR))
        moved = int(np.count_nonzero(can_move))
        if moved:
            below_top[can_move] = bottom[can_move]
            bottom[can_move] = int(Material.AIR)
            below.dirty = True
            chunk.dirty = True

    if above is not None:
        top = chunk.data[:, :, size - 1]
        above_bottom = above.data[:, :, 0]
        movable = np.isin(top, [int(mat) for mat in MOVES_UP])
        can_move = movable & (above_bottom == int(Material.AIR))
        moved_up = int(np.count_nonzero(can_move))
        if moved_up:
            above_bottom[can_move] = top[can_move]
            top[can_move] = int(Material.AIR)
            above.dirty = True
            chunk.dirty = True
        moved += moved_up

    if left is not None:
        face = chunk.data[0, :, :]
        target = left.data[size - 1, :, :]
        movable = np.isin(face, [int(Material.LIQUID), int(Material.GAS), int(Material.GRANULAR)])
        can_move = movable & (target == int(Material.AIR))
        moved_side = int(np.count_nonzero(can_move))
        if moved_side:
            target[can_move] = face[can_move]
            face[can_move] = int(Material.AIR)
            left.dirty = True
            chunk.dirty = True
        moved += moved_side

    if right is not None:
        face = chunk.data[size - 1, :, :]
        target = right.data[0, :, :]
        movable = np.isin(face, [int(Material.LIQUID), int(Material.GAS), int(Material.GRANULAR)])
        can_move = movable & (target == int(Material.AIR))
        moved_side = int(np.count_nonzero(can_move))
        if moved_side:
            target[can_move] = face[can_move]
            face[can_move] = int(Material.AIR)
            right.dirty = True
            chunk.dirty = True
        moved += moved_side

    if back is not None:
        face = chunk.data[:, 0, :]
        target = back.data[:, size - 1, :]
        movable = np.isin(face, [int(Material.LIQUID), int(Material.GAS), int(Material.GRANULAR)])
        can_move = movable & (target == int(Material.AIR))
        moved_side = int(np.count_nonzero(can_move))
        if moved_side:
            target[can_move] = face[can_move]
            face[can_move] = int(Material.AIR)
            back.dirty = True
            chunk.dirty = True
        moved += moved_side

    if front is not None:
        face = chunk.data[:, size - 1, :]
        target = front.data[:, 0, :]
        movable = np.isin(face, [int(Material.LIQUID), int(Material.GAS), int(Material.GRANULAR)])
        can_move = movable & (target == int(Material.AIR))
        moved_side = int(np.count_nonzero(can_move))
        if moved_side:
            target[can_move] = face[can_move]
            face[can_move] = int(Material.AIR)
            front.dirty = True
            chunk.dirty = True
        moved += moved_side

    return moved


def step_world(
    chunks: Dict[ChunkCoord, Chunk],
    config: SimulationConfig | None = None,
    rng: np.random.Generator | None = None,
) -> StepStats:
    """Advance simulation for all chunks."""

    if config is None:
        config = SimulationConfig()
    if rng is None:
        rng = np.random.default_rng(config.rng_seed)

    moved_total = 0
    touched = 0
    updates: Dict[ChunkCoord, np.ndarray] = {}

    for coord, chunk in chunks.items():
        updated, moved = step_chunk_vertical(chunk, rng, config)
        updates[coord] = updated
        if moved:
            moved_total += moved
            touched += 1

    for coord, data in updates.items():
        chunk = chunks[coord]
        if not np.array_equal(chunk.data, data):
            chunk.data = data
            chunk.dirty = True

    lateral_updates: Dict[ChunkCoord, np.ndarray] = {}
    for coord, chunk in chunks.items():
        updated, moved = step_chunk_lateral(chunk, rng, config)
        lateral_updates[coord] = updated
        if moved:
            moved_total += moved
            touched += 1

    for coord, data in lateral_updates.items():
        chunk = chunks[coord]
        if not np.array_equal(chunk.data, data):
            chunk.data = data
            chunk.dirty = True

    for coord in list(chunks.keys()):
        moved_total += apply_boundary_transfers(chunks, coord)

    return StepStats(moved_voxels=moved_total, touched_chunks=touched)
