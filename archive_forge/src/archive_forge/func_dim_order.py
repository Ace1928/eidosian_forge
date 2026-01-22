import copyreg
import enum
import functools
import warnings
from collections import OrderedDict
from copy import deepcopy
from numbers import Number
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch._C as _C
import torch.utils.hooks as hooks
from torch._namedtensor_internals import (
from torch.overrides import (
from torch.utils.dlpack import DLDeviceType
def dim_order(self):
    """

        dim_order() -> tuple

        Returns a tuple of int describing the dim order or physical layout of :attr:`self`.

        Args:
            None

        Dim order represents how dimensions are laid out in memory,
        starting from the outermost to the innermost dimension.

        Example::
            >>> torch.empty((2, 3, 5, 7)).dim_order()
            (0, 1, 2, 3)
            >>> torch.empty((2, 3, 5, 7), memory_format=torch.channels_last).dim_order()
            (0, 2, 3, 1)

        .. warning::
            The dim_order tensor API is experimental and subject to change.

        """
    if has_torch_function_unary(self):
        return handle_torch_function(Tensor.dim_order, (self,), self)
    import torch._prims_common as utils
    return tuple(utils.compute_elementwise_output_logical_to_physical_perm(self))