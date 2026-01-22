import itertools
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.fast_repartition import fast_repartition
from ray.data._internal.plan import AllToAllStage
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.shuffle_and_partition import (
from ray.data._internal.sort import SortKey, sort_impl
from ray.data._internal.split import _split_at_index, _split_at_indices
from ray.data.block import (
from ray.data.context import DataContext
def do_zip_all(block_list: BlockList, clear_input_blocks: bool, *_):
    base_block_list = block_list
    base_blocks_with_metadata = block_list.get_blocks_with_metadata()
    base_block_rows, base_block_bytes = _calculate_blocks_rows_and_bytes(base_blocks_with_metadata)
    other_block_list = other._plan.execute(preserve_order=True)
    other_blocks_with_metadata = other_block_list.get_blocks_with_metadata()
    other_block_rows, other_block_bytes = _calculate_blocks_rows_and_bytes(other_blocks_with_metadata)
    inverted = False
    if sum(other_block_bytes) > sum(base_block_bytes):
        base_block_list, other_block_list = (other_block_list, base_block_list)
        base_blocks_with_metadata, other_blocks_with_metadata = (other_blocks_with_metadata, base_blocks_with_metadata)
        base_block_rows, other_block_rows = (other_block_rows, base_block_rows)
        inverted = True
    indices = list(itertools.accumulate(base_block_rows))
    indices.pop(-1)
    total_base_rows = sum(base_block_rows)
    total_other_rows = sum(other_block_rows)
    if total_base_rows != total_other_rows:
        raise ValueError(f'Cannot zip datasets of different number of rows: {total_base_rows}, {total_other_rows}')
    aligned_other_blocks_with_metadata = _split_at_indices(other_blocks_with_metadata, indices, other_block_list._owned_by_consumer, other_block_rows)
    del other_blocks_with_metadata
    base_blocks = [b for b, _ in base_blocks_with_metadata]
    other_blocks = aligned_other_blocks_with_metadata[0]
    del base_blocks_with_metadata, aligned_other_blocks_with_metadata
    if clear_input_blocks:
        base_block_list.clear()
        other_block_list.clear()
    do_zip = cached_remote_fn(_do_zip, num_returns=2)
    out_blocks = []
    out_metadata = []
    for base_block, other_blocks in zip(base_blocks, other_blocks):
        res, meta = do_zip.remote(base_block, *other_blocks, inverted=inverted)
        out_blocks.append(res)
        out_metadata.append(meta)
    del base_blocks, other_blocks
    out_metadata = ray.get(out_metadata)
    blocks = BlockList(out_blocks, out_metadata, owned_by_consumer=base_block_list._owned_by_consumer)
    return (blocks, {})