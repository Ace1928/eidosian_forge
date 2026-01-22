import weakref
from typing import Any, Callable, List, Optional
import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.optim.zero_redundancy_optimizer import (
from torch.nn.parallel.distributed import DistributedDataParallel
def hook_with_zero_interleaved_fn(state, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    """
        Returns a :class:`Future` that gives a gradient bucket tensor and
        performs a partial :class:`ZeroRedundancyOptimizer` :meth:`step` using
        the gradients in that bucket.
        Arguments:
            state: any state for the hook.
            bucket (dist.GradBucket): the :class:`DistributedDataParallel`
                gradient bucket.
        """
    fut = hook(state, bucket)
    _hook_with_zero_step_setup(ddp_ref, zero, bucket)
    if zero._overlap_info.status != _OverlapStatus.INITIALIZED:
        return fut

    def zero_step(fut: torch.futures.Future) -> torch.Tensor:
        """
            Performs a partial :class:`ZeroRedundancyOptimizer` :meth:`step`
            using the gradients in the given :class:`DistributedDataParallel`
            gradient bucket.

            Returns:
                A :class:`torch.Tensor` representing the contents of the
                gradient bucket.
            """
        overlap_info = zero._overlap_info
        bucket_index = bucket.index()
        rank = zero.global_rank
        assigned_ranks = overlap_info.assigned_ranks_per_bucket[bucket_index]
        overlap_info.bucket_indices_seen.append(bucket_index)
        if rank in assigned_ranks:
            _perform_local_step(bucket, zero, rank)
        _broadcast_bucket(bucket_index, zero)
        num_buckets = len(overlap_info.params_per_bucket)
        if len(overlap_info.bucket_indices_seen) == num_buckets:
            overlap_info.wait_for_broadcasts()
            overlap_info.clear_per_iter_info()
        return bucket.buffer()
    return fut.then(zero_step)