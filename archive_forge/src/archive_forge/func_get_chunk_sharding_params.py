from typing import List, Optional, Tuple
from torch.distributed._shard.metadata import ShardMetadata
def get_chunk_sharding_params(sharding_dim_size, world_size, spec, rank):
    """
    Generate the start pos and offset length for the current rank for
    chunk sharding.

    Args:
        sharding_dim_size(int): The dimension length which we shard on.
        world_size(int): number of ranks.
        spec (:class:`torch.distributed._shard.sharding_spec.ChunkShardingSpec`):
            sharding spec.
        rank(int): # of cuda process.

    Returns:
        start_pos(int): start position of sharded tensor on the given rank.
        chunk_size(int): chunk size of sharded tensor on the given rank.
    """
    split_size = get_split_size(sharding_dim_size, world_size)
    current_offsets = 0
    start_pos = current_offsets
    for idx, placement in enumerate(spec.placements):
        chunk_size = get_chunked_dim_size(sharding_dim_size, split_size, idx)
        if rank == placement.rank():
            start_pos = current_offsets
            break
        current_offsets += chunk_size
    return (start_pos, chunk_size)