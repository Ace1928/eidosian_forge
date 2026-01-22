import functools
import os
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple
import torch
from torch import Tensor
import torch.distributed as dist
@torch.no_grad()
def reduce_scatter_async(self, input_list: List[Tensor], group: 'ProcessGroup', callback_fn: Optional[Callable]=None) -> None:
    """
        Reduce-scatter a list of tensors asynchronously, so smaller reductions
        can be bucketed together. The given callback (``callback_fn``) will be
        called with the reduced result at some later time. Call ``flush()`` to
        force all queued ops and callbacks to be executed.

        Note that large inputs will be reduced immediately, and this function
        may also flush the relevant bucket to make room for ``input_list``.

        Args:
            input_list (List[Tensor]): list of tensors to reduce-scatter. List
                should contain ``group.size()`` tensors and each tensor should
                have identical shape, dtype and device.
            group (ProcessGroup): process group for reduction
            callback_fn (Callable, Optional): callback function to call after
                the reduction executes. Function will be called with a single
                argument corresponding to the reduced result.
        """
    world_size = group.size()
    assert len(input_list) == world_size, f'reduce_scatter received {len(input_list)} inputs, expected group.size() ({world_size})'
    first_input = input_list[0]
    first_input_size = first_input.numel()
    bucket_shard_size = self._get_shard_size(first_input.element_size(), world_size)
    if first_input_size > bucket_shard_size:
        output = torch.zeros_like(input_list[0])
        if hasattr(dist, '_reduce_scatter_base') and enable_nccl_base_collectives:
            input_flattened = torch.cat(input_list)
            dist._reduce_scatter_base(output, input_flattened, group=group)
        else:
            dist.reduce_scatter(output, input_list, group=group)
        if callback_fn is not None:
            callback_fn(output)
        return
    bucket = self._get_bucket(first_input, group)
    if first_input_size > bucket.data.size(1) - bucket.offset:
        bucket.flush()
    stacked_input = torch.stack(input_list).view(world_size, first_input_size)
    offset = bucket.offset
    bucket.data[:, offset:offset + first_input_size].copy_(stacked_input)
    bucket.offset += first_input_size
    if callback_fn is not None:
        result_view = bucket.output_shard[offset:offset + first_input_size].view_as(first_input)
        bucket.callbacks.append(functools.partial(callback_fn, result_view))