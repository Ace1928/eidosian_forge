import contextlib
import copy
from enum import Enum, auto
import functools
import logging
from math import inf
import os
import time
import traceback
import typing
from typing import (
import torch
from torch.autograd import Variable
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from fairscale.internal.containers import apply_to_tensors
from fairscale.internal.parallel import (
from fairscale.internal.params import calc_grad_norm, recursive_copy_to_device
from fairscale.internal.reduce_scatter_bucketer import ReduceScatterBucketer
from fairscale.internal.state_dict import replace_by_prefix_
from fairscale.nn.misc import FlattenParamsWrapper, _enable_pre_load_state_dict_hook
from fairscale.nn.wrap import auto_wrap, config_auto_wrap_policy, enable_wrap
from . import fsdp_optim_utils as ou
def gather_full_optim_state_dict(self, optim: torch.optim.Optimizer, **ignored: Dict) -> Optional[Dict[str, Any]]:
    """Return the last known global optimizer state. The returned state is compatible with Pytorch, in that the
        sharded properties are not exposed. Multiple parameter groups are not yet supported.

        This should be called only on the root FSDP instance.
        Nested FSDP instances are supported as long as they have the same world_size as the parent or world_size=1.

        Args:
            optim (Optimizer): an optimizer instance for this FSDP rank. Its state_dict is
                        used in the consolidation. However, its state is not modified.

        Returns:

            * A dict with four entries (On rank zero, other workers return ``None``)
                * state - a dict holding gathered optimization state, 1 entry per unflat parameter
                * param_groups - a dict containing the 1 parameter group
                * param_id_map - global (unflat) to local (flat) id mapping
                * uncollected_local_ids - keys in the state dict that were not broadcast

        """
    if not self.flatten_parameters:
        raise NotImplementedError('optim state dict requires flatten_parameters=True')
    self._lazy_init()
    sd = self._remove_uncollectable_params_from_optim_state_dict(optim.state_dict())
    assert {'param_groups', 'state'}.issubset(set(sd.keys())), f'{set(sd.keys())} not a superset of {('param_groups', 'state')}'
    assert len(sd['param_groups']) == 1, 'Param groups are not supported'
    state, singleton_state = self._gather_optim_state(sd.pop('state'))
    pad_info = self._broadcast_pad_info_to_r0()
    if self.rank != 0:
        return None
    new_state_dict = ou.build_unflat_state_dict(get_fsdp_instances(self, skip_empty=True), pad_info, state, singleton_state, self.uncollected_opt_state, sd)
    self.uncollected_opt_state = {}
    assert 'uncollected_local_ids' in new_state_dict
    return new_state_dict