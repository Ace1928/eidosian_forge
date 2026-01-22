import logging
import os
from typing import List
import numpy as np
import ray
from ray.util.collective import types
def reducescatter_multigpu(output_tensor_list, input_tensor_lists, group_name: str='default', op=types.ReduceOp.SUM):
    """Reducescatter a list of tensors across all GPUs.

    Args:
        output_tensor_list: the resulted list of tensors, with
            shape: num_gpus * shape(tensor).
        input_tensor_lists: the original tensors, with shape:
            num_gpus * world_size * shape(tensor).
        group_name: the name of the collective group.
        op: The reduce operation.

    Returns:
        None.
    """
    if not types.cupy_available():
        raise RuntimeError('Multigpu calls requires NCCL and Cupy.')
    _check_tensor_lists_input(input_tensor_lists)
    _check_tensor_list_input(output_tensor_list)
    g = _check_and_get_group(group_name)
    opts = types.ReduceScatterOptions()
    opts.reduceOp = op
    g.reducescatter(output_tensor_list, input_tensor_lists, opts)