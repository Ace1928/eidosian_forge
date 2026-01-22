from typing import List, Optional, Tuple
from torch.distributed._shard.metadata import ShardMetadata
def check_tensor(shards_metadata, tensor_dims) -> None:
    """
    Checks if the shards_metadata is compatible with the provided tensor dims.

    Args:
        shards_metadata(List[ShardMetadata]): List of :class:`ShardMetadata`
            objects representing each shard of the tensor.
        tensor_dims(Sequence of int): Dimensions of tensor to verify
    Raises:
        ``ValueError`` if not compatible.
    """
    tensor_rank = len(tensor_dims)
    shards_rank = len(shards_metadata[0].shard_offsets)
    if tensor_rank != shards_rank:
        raise ValueError(f'Rank of tensor is {tensor_rank}, but shards rank is {shards_rank}')
    total_shard_volume = 0
    for shard in shards_metadata:
        shard_volume = 1
        for i, shard_length in enumerate(shard.shard_sizes):
            shard_volume *= shard_length
            if shard.shard_offsets[i] + shard.shard_sizes[i] > tensor_dims[i]:
                raise ValueError(f'Shard offset {shard.shard_offsets[i]} and length {shard.shard_sizes[i]} exceeds tensor dim: {tensor_dims[i]} for shard {shard}')
        total_shard_volume += shard_volume
    tensor_volume = 1
    for size in tensor_dims:
        tensor_volume *= size
    if total_shard_volume != tensor_volume:
        raise ValueError(f'Total volume of shards: {total_shard_volume} does not match tensor volume: {tensor_volume}, in other words all the individual shards do not cover the entire tensor')