import collections.abc
import copy
from typing import Optional, List, Sequence
import torch
from torch.distributed import distributed_c10d
from torch.distributed import rpc
from torch.distributed._shard.sharding_spec._internals import (
from torch.distributed._shard.metadata import ShardMetadata
from .metadata import TensorProperties, ShardedTensorMetadata
from .shard import Shard
def build_global_metadata(gathered_metadatas: Sequence[Optional[ShardedTensorMetadata]]):
    global_sharded_tensor_metadata = None
    global_metadata_rank = 0
    for rank, rank_metadata in enumerate(gathered_metadatas):
        if rank_metadata is None:
            continue
        if global_sharded_tensor_metadata is None:
            global_sharded_tensor_metadata = copy.deepcopy(rank_metadata)
            global_metadata_rank = rank
        else:
            _raise_if_mismatch(global_sharded_tensor_metadata.size, rank_metadata.size, 'global_size', [global_metadata_rank, rank], is_local=False)
            _raise_if_mismatch(global_sharded_tensor_metadata.tensor_properties.dtype, rank_metadata.tensor_properties.dtype, 'dtype', [global_metadata_rank, rank], is_local=False)
            _raise_if_mismatch(global_sharded_tensor_metadata.tensor_properties.requires_grad, rank_metadata.tensor_properties.requires_grad, 'requires_grad', [global_metadata_rank, rank], is_local=False)
            _raise_if_mismatch(global_sharded_tensor_metadata.tensor_properties.pin_memory, rank_metadata.tensor_properties.pin_memory, 'pin_memory', [global_metadata_rank, rank], is_local=False)
            global_sharded_tensor_metadata.shards_metadata.extend(rank_metadata.shards_metadata)
    if global_sharded_tensor_metadata is not None:
        validate_non_overlapping_shards_metadata(global_sharded_tensor_metadata.shards_metadata)
        check_tensor(global_sharded_tensor_metadata.shards_metadata, global_sharded_tensor_metadata.size)
    else:
        raise ValueError('ShardedTensor have no local shards on all ranks!')
    return global_sharded_tensor_metadata