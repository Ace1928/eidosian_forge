import builtins
import torch
from torch.distributed._shard.sharding_spec import (
from torch.distributed._shard.sharding_spec._internals import (
def generate_local_weight_sharding_params_for_test(local_weight, sharded_dim, gpu_num, spec, rank):
    """
    Shard the local weight based the given spec, so we can compare against
    the one from sharded tensor.

    Args:
        local_weight: weight matrix to be sharded.
        sharded_dim: The dimension which we shard on.
        gpu_num: number of ranks.
        spec: sharding spec.
        rank: # of cuda process.

    Returns:
        start_pos: start position of sharded weight on the given rank.
        chunk_size: chunk size of sharded weight on the given rank.
    """
    sharding_dim_size = local_weight.size(sharded_dim)
    split_size = get_split_size(sharding_dim_size, gpu_num)
    current_offsets = 0
    start_pos = current_offsets
    for idx, placement in enumerate(spec.placements):
        chunk_size = get_chunked_dim_size(sharding_dim_size, split_size, idx)
        if rank == placement.rank():
            start_pos = current_offsets
            break
        current_offsets += chunk_size
    return (start_pos, chunk_size)