from typing import cast, List
import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op
from torch.distributed.nn.functional import all_gather, reduce_scatter
from ._common import (
@custom_sharding_spec_op(ChunkShardingSpec, torch.nn.functional.embedding_bag)
def sharded_embedding_bag(types, args, kwargs, pg):
    """
    Handles ``__torch_function__`` dispatch for ``torch.nn.functional.embedding_bag``.
    This method computes a sharded embedding bag aggregation and has the following limitations:

    1. Supports only sharding of ``weight``.
    2. Supports only ``ChunkShardingSpec``.
    3. Supports only a single local shard per rank.
    4. Supports all specs except for scale_grad_by_freq, sparse, etc.

    Based on the dimension that the weight is sharded on, there are two
    algorithms:

    ROWWISE SHARDING
    ================
    For row-wise sharding the weight is sharded on dimension 0.

    The overall algorithm can be best explained with an example. Let's assume
    the dims for input are (4 x 6) and W are (16 x 17) and W is sharded across
    4 GPUs creating 4 shard of (4 x 17).
    The algorithm is as follows:

    1. First the input is all gathered to all ranks, since this is SPMD and
       input is actually sharded across all ranks. The inputs then become a
       4 (4 x 6) tensor on each rank. For example if the given input is
       tensor([[6, 5, 2, 9, 6, 3],
               [3, 1, 2, 4, 7, 6],
               [4, 0, 4, 9, 8, 9],
               [8, 6, 6, 4, 6, 1]])
       on rank 0.
       Then on every rank, we will have this tensor.
       If input itself is already replicated, no all-gather will be done.
    2. Next, we mask the ID which are not stored on that rank.
       For example on rank 0, we store ID [0, 1, 2]. We only keep the ID
       inside the set of numbers. The rest of them will be masked to an extra row.
       The masked matrix will be used for embedding look up and is like:
       tensor([[4, 4, 2, 4, 4, 4],
               [4, 1, 2, 4, 4, 4],
               [4, 0, 4, 4, 4, 4],
               [4, 4, 4, 4, 4, 1]])
    3. If ``max_norm`` is specified, the extra row guarantees that the mask ID will
       not affect the behavior of weigh re-norm.
    4. The example above only happens in one rank and each rank does a very similar thing.
       For "Mean" mode we need to divide by either column size (2D) or the interval length
       defined by the offset (excluding the row specified in ``padding_idx``).
       We also need to mask the unexisting row to neg Inf so that negative value does not
       gets wiped out in the "Max" mode.

    COLWISE SHARDING
    ================
    For col-wise sharding the weight is sharded on dimension 1.

    The overall algorithm can be best explained with an example. Let's assume
    the dims for input are (4 x 6) and W are (16 x 17) and W is sharded across
    4 GPUs creating 3 shards of (16 x 5) and 1 shard of (16 x 2).
    The algorithm is as follows:

    1. First the input is broadcasted to all ranks, since this is SPMD we
       actually do an all_gather for all the inputs resulting in 4 (4 x 6)
       inputs on each rank.
    2. Next we perform local embedding bag operation under the given mode by
       apply each input (4 x 6) with the local shard (16 x 5) ((16 x 2) for the last).
       This results in 4 (5 x 4) ((2 x 4) for the last) matrices on each rank.
       We transpose the aggregation result.
    3. Next, we concatenate these 4 matrices and perform an all2all to share the
       appropriate (5 x 4) or (2 x 4) matrices to each rank.
    4. Now, each rank receives a (17 x 4) matrix which is basically the
       size of the result we need.
    5. If placements are not in order any appropriate rearrangement of columns
       are done for the (17 x 4) matrix and finally we transpose the output again.
    6. If max_norm is specified, we manually sum up the norm and renorm. Because
       the renorm must be in place, we need to override the local_shard to mimic
       this behavior.
    """
    _validate_embedding_bag_param(args, kwargs)
    input = args[0]
    weight = args[1]
    offsets = kwargs.get('offsets')
    per_sample_weights = kwargs.get('per_sample_weights')
    mode = kwargs.get('mode')
    max_norm = kwargs.get('max_norm')
    norm_type = kwargs.get('norm_type')
    include_last_offset = kwargs.get('include_last_offset')
    padding_idx = kwargs.get('padding_idx')
    local_shard = weight.local_tensor().contiguous()
    sharding_dim = weight._sharding_spec.dim
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)
    if include_last_offset:
        offsets = offsets[:-1]
    if sharding_dim == 1:
        output, local_shard = _handle_col_wise_sharding(input, world_size, weight, local_shard, offsets, per_sample_weights, mode, max_norm, norm_type, padding_idx, pg)
        weight.local_shards()[0].tensor = local_shard
        return output
    elif sharding_dim == 0:
        return _handle_row_wise_sharding(input, world_size, weight, local_shard, offsets, per_sample_weights, mode, max_norm, norm_type, padding_idx, rank, pg)
    else:
        raise RuntimeError(f'nn.EmbeddingBag weight sharded on dim {sharding_dim} not supported!')