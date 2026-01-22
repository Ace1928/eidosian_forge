from collections import OrderedDict, namedtuple
import itertools
import warnings
import functools
import weakref
import torch
from torch._prims_common import DeviceLikeType
from ..parameter import Parameter
import torch.utils.hooks as hooks
from torch import Tensor, device, dtype
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
from typing_extensions import Self
from ...utils.hooks import RemovableHandle
def compute_should_use_set_data(tensor, tensor_applied):
    if torch._has_compatible_shallow_copy_type(tensor, tensor_applied):
        return not torch.__future__.get_overwrite_module_params_on_conversion()
    else:
        return False