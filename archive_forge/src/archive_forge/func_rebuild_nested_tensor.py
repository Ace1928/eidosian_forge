import multiprocessing
import os
import threading
from multiprocessing.reduction import ForkingPickler
from multiprocessing.util import register_after_fork
from typing import Union
import torch
import torch.utils.hooks
from torch._namedtensor_internals import check_serializing_named_tensor
def rebuild_nested_tensor(rebuild_buffer_func, rebuild_buffer_args, rebuild_sizes_func, rebuild_sizes_args, rebuild_strides_func, rebuild_strides_args, rebuild_offsets_func, rebuild_offsets_args):
    buffer = rebuild_buffer_func(*rebuild_buffer_args)
    sizes = rebuild_sizes_func(*rebuild_sizes_args)
    strides = rebuild_strides_func(*rebuild_strides_args)
    offsets = rebuild_offsets_func(*rebuild_offsets_args)
    return torch._nested_view_from_buffer_copy(buffer, sizes, strides, offsets)