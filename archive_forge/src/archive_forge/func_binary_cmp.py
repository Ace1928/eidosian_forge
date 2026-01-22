import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as distributed_c10d
from torch.distributed._shard.sharded_tensor import (
def binary_cmp(cmp_fun, types, args, kwargs=None, process_group=None):
    if len(args) != 2:
        raise ValueError(f'Expected two arguments for torch.{cmp_fun.__name__}')
    result = True
    st1 = args[0]
    st2 = args[1]
    if not (isinstance(st1, ShardedTensor) and isinstance(st2, ShardedTensor)):
        raise TypeError(f'Both arguments to torch.{cmp_fun.__name__} need to be of type ShardedTensor')
    if st1._process_group != st2._process_group:
        return False
    if distributed_c10d._rank_not_in_group(st1._process_group) or distributed_c10d._rank_not_in_group(st2._process_group):
        return distributed_c10d._rank_not_in_group(st1._process_group) == distributed_c10d._rank_not_in_group(st2._process_group)
    if st1.metadata() != st2.metadata():
        return _communicate_result(False, st1._process_group)
    st1_local_shards = st1.local_shards()
    st2_local_shards = st2.local_shards()
    if len(st1_local_shards) != len(st2_local_shards):
        return _communicate_result(False, st1._process_group)
    if kwargs is None:
        kwargs = {}
    for idx in range(len(st1_local_shards)):
        if st1_local_shards[idx].metadata != st2_local_shards[idx].metadata:
            return _communicate_result(False, st1._process_group)
        if not cmp_fun(st1_local_shards[idx].tensor, st2_local_shards[idx].tensor, **kwargs):
            return _communicate_result(False, st1._process_group)
    return _communicate_result(True, st1._process_group)