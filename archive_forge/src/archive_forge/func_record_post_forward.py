import itertools
import warnings
from enum import auto, Enum
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed.fsdp._common_utils import _FSDPState, _get_param_to_fqns
from torch.distributed.fsdp._flat_param import FlatParamHandle
def record_post_forward(self, handle: Optional[FlatParamHandle]) -> None:
    """
        Records ``handles`` in the post-forward order, where ``handles`` should
        be a group of handles used in the same module's forward. If ``handles``
        is empty, then it is omitted.

        Unlike :meth:`record_pre_forward`, this records the order *every*
        iteration with the expectation that the recorded order is reset in
        :meth:`next_iter`.
        """
    if not handle:
        return
    if handle._post_forward_index:
        self.handles_post_forward_order.append(handle)
        return
    index = len(self.handles_post_forward_order)
    handle._post_forward_index = index
    self.handles_post_forward_order.append(handle)