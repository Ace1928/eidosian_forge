import torch
from torch import Tensor
import itertools
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from torch.utils import _pytree as pytree
from functools import partial
from torch.utils._mode_utils import no_dispatch, all_same_mode
import torch.autograd.forward_ad as fwAD
from typing import Callable
import re
def check_metadata_consistency(wrapper_tensor, CCT):
    if not isinstance(wrapper_tensor, CCT):
        return
    things_to_check = {'shape': Tensor.size, 'dtype': lambda x: x.dtype, 'device': lambda x: x.device, 'numel': Tensor.numel, 'stride': Tensor.stride, 'storage_offset': Tensor.storage_offset}
    for metadata_name, metadata_accessor in things_to_check.items():
        check_attr_consistency(wrapper_tensor, metadata_name, metadata_accessor)