import weakref
from typing import Any, Callable, List, Optional
import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.optim.zero_redundancy_optimizer import (
from torch.nn.parallel.distributed import DistributedDataParallel
def hook_with_zero_step(hook: Callable[[Any, dist.GradBucket], torch.futures.Future], ddp: DistributedDataParallel, zero: ZeroRedundancyOptimizer, shard_buckets: bool=False) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    """
    Modifies the given ``hook`` to overlap the :class:`ZeroRedundancyOptimizer`
    optimizer step with the :class:`DistributedDataParallel` backward pass,
    where the optimizer step computation begins after the last gradient bucket
    computation has finished.

    This approach overlaps the optimizer computation and communication with the
    backward communication. In particular, the backward computation proceeds
    contiguously, and the optimizer computation follows, overlapping with
    outstanding backward communication (i.e. all-reduces) and possibly other
    optimizer communication (i.e. broadcasts).

    This approach may be preferred over :meth:`hook_with_zero_step_interleaved`
    if communication is relatively slow compared to computation.

    Arguments:
        hook (Callable[[Any, dist.GradBucket], torch.futures.Future]): the hook
            to modify.
        ddp (DistributedDataParallel): the :class:`DistributedDataParallel`
            instance to use.
        zero (ZeroRedundancyOptimizer): the :class:`ZeroRedundancyOptimizer`
            instance to use.
        shard_buckets (bool): if ``True``, then the assignment of each
            :class:`DistributedDataParallel` bucket is partitioned across
            possibly multiple :class:`ZeroRedundancyOptimizer` instances (i.e.
            across possibly multiple ranks) to approximate uniformity; if
            ``False``, then each bucket is wholly assigned to a single
            :class:`ZeroRedundancyOptimizer` instance (i.e. to a single rank).

    Returns:
        The modified hook.

    Raises:
        ValueError: if ``zero`` was constructed with ``overlap_with_ddp=False``.
        RuntimeError: if using any backend other than NCCL/HCCL since currently
            Gloo may hang.

    .. warning::
        Given the way that overlapping :class:`DistributedDataParallel` with
        :class:`ZeroRedundancyOptimizer` is currently implemented, the first
        two or three training iterations do not perform parameter updates in
        the optimizer step, depending on if ``static_graph=False`` or
        ``static_graph=True``, respectively. This is because it needs
        information about the gradient bucketing strategy used by
        :class:`DistributedDataParallel`, which is not finalized until the
        second forward pass if ``static_graph=False`` or until the third
        forward pass if ``static_graph=True``.
    """
    if not zero._overlap_with_ddp:
        raise ValueError('ZeroRedundancyOptimizer must be constructed with `overlap_with_ddp=True` to use this hook properly')
    ddp_ref = weakref.ref(ddp)
    pg = dist.get_backend(ddp_ref().process_group)
    if pg != dist.Backend.NCCL and pg != 'hccl':
        raise RuntimeError('Overlapping DDP with ZeRO using this approach currently requires NCCL/HCCL backend to avoid hangs')
    if shard_buckets:
        zero._overlap_info.shard_buckets = True
        zero._overlap_info.total_size = 0

    def hook_with_zero_fn(state: Any, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
        """
        Returns a :class:`Future` that gives a gradient bucket tensor and
        performs the equivalent of a :class:`ZeroRedundancyOptimizer`
        :meth:`step` if ``bucket`` is the last gradient bucket.

        The function performs additional computation on the iteration that
        the :class:`DistributedDataParallel` buckets are rebuilt to collect
        information used to implement the modified hook.

        Arguments:
            state (Any): any state for the hook.
            bucket (dist.GradBucket): the :class:`DistributedDataParallel`
                gradient bucket.
        """
        fut = hook(state, bucket)
        _hook_with_zero_step_setup(ddp_ref, zero, bucket)
        if zero._overlap_info.status != _OverlapStatus.INITIALIZED:
            return fut
        overlap_info = zero._overlap_info
        bucket_index = bucket.index()
        rank = zero.global_rank
        assert overlap_info.status == _OverlapStatus.INITIALIZED
        assert len(overlap_info.assigned_ranks_per_bucket) > bucket_index, '`assigned_ranks_per_bucket` is not fully constructed'
        assigned_to_bucket = rank in overlap_info.assigned_ranks_per_bucket[bucket_index]
        if assigned_to_bucket:
            overlap_info.bucket_index_to_bucket[bucket_index] = bucket
            overlap_info.bucket_index_to_future[bucket_index] = fut
        if len(overlap_info.bucket_indices_seen) > 0:
            assert overlap_info.bucket_indices_seen[-1] == bucket_index - 1, 'Bucket indices are not in incremental order'
        else:
            assert bucket_index == 0, 'Bucket indices do not start from 0'
        overlap_info.bucket_indices_seen.append(bucket_index)
        num_buckets = len(overlap_info.params_per_bucket)
        is_last_bucket = bucket_index == num_buckets - 1
        if not is_last_bucket:
            return fut
        for bucket_index in range(num_buckets):
            assigned_ranks = overlap_info.assigned_ranks_per_bucket[bucket_index]
            if rank in assigned_ranks:
                assert bucket_index in overlap_info.bucket_index_to_future, f'All-reduce future for bucket {bucket_index} not saved on rank {rank}'
                allreduce_future = overlap_info.bucket_index_to_future[bucket_index]
                allreduce_future.wait()
                curr_bucket = overlap_info.bucket_index_to_bucket[bucket_index]
                _perform_local_step(curr_bucket, zero, rank)
            _broadcast_bucket(bucket_index, zero)
        overlap_info.wait_for_broadcasts()
        overlap_info.clear_per_iter_info()
        return fut
    return hook_with_zero_fn