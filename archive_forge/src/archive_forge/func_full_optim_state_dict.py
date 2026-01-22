import contextlib
import copy
import functools
import math
import traceback
import warnings
from contextlib import contextmanager
from enum import auto, Enum
from typing import (
import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
from torch.distributed.algorithms._comm_hooks import LOW_PRECISION_HOOKS
from torch.distributed.fsdp._common_utils import (
from torch.distributed.fsdp._dynamo_utils import _annotate_modules_for_dynamo
from torch.distributed.fsdp._init_utils import (
from torch.distributed.fsdp._runtime_utils import (
from torch.distributed.fsdp._wrap_utils import _auto_wrap
from torch.distributed.fsdp.api import (
from torch.distributed.utils import _p_assert
from ._flat_param import FlatParameter
from ._optim_utils import (
from ._state_dict_utils import _register_all_state_dict_hooks
from ._unshard_param_utils import (
from .wrap import CustomPolicy, ModuleWrapPolicy
@staticmethod
def full_optim_state_dict(model: torch.nn.Module, optim: torch.optim.Optimizer, optim_input: Optional[Union[List[Dict[str, Any]], Iterable[torch.nn.Parameter]]]=None, rank0_only: bool=True, group: Optional[dist.ProcessGroup]=None) -> Dict[str, Any]:
    """Return the full optimizer state-dict.

        Consolidates the full optimizer state on rank 0 and returns it
        as a :class:`dict` following the convention of
        :meth:`torch.optim.Optimizer.state_dict`, i.e. with keys ``"state"``
        and ``"param_groups"``. The flattened parameters in ``FSDP`` modules
        contained in ``model`` are mapped back to their unflattened parameters.

        .. warning:: This needs to be called on all ranks since it uses
            collective communications. However, if ``rank0_only=True``, then
            the state dict is only populated on rank 0, and all other ranks
            return an empty :class:`dict`.

        .. warning:: Unlike ``torch.optim.Optimizer.state_dict()``, this method
            uses full parameter names as keys instead of parameter IDs.

        .. note:: Like in :meth:`torch.optim.Optimizer.state_dict`, the tensors
            contained in the optimizer state dict are not cloned, so there may
            be aliasing surprises. For best practices, consider saving the
            returned optimizer state dict immediately, e.g. using
            ``torch.save()``.

        Args:
            model (torch.nn.Module): Root module (which may or may not be a
                :class:`FullyShardedDataParallel` instance) whose parameters
                were passed into the optimizer ``optim``.
            optim (torch.optim.Optimizer): Optimizer for ``model`` 's
                parameters.
            optim_input (Optional[Union[List[Dict[str, Any]], Iterable[torch.nn.Parameter]]]):
                Input passed into the optimizer ``optim`` representing either a
                :class:`list` of parameter groups or an iterable of parameters;
                if ``None``, then this method assumes the input was
                ``model.parameters()``. This argument is deprecated, and there
                is no need to pass it in anymore. (Default: ``None``)
            rank0_only (bool): If ``True``, saves the populated :class:`dict`
                only on rank 0; if ``False``, saves it on all ranks. (Default:
                ``True``)
            group (dist.ProcessGroup): Model's process group or ``None`` if using
                the default process group. (Default: ``None``)

        Returns:
            Dict[str, Any]: A :class:`dict` containing the optimizer state for
            ``model`` 's original unflattened parameters and including keys
            "state" and "param_groups" following the convention of
            :meth:`torch.optim.Optimizer.state_dict`. If ``rank0_only=True``,
            then nonzero ranks return an empty :class:`dict`.
        """
    FullyShardedDataParallel._warn_legacy_optim_state_dict('full_optim_state_dict', 'optim_state_dict')
    return FullyShardedDataParallel._optim_state_dict_impl(model=model, optim=optim, optim_state_dict=optim.state_dict(), optim_input=optim_input, rank0_only=rank0_only, group=group, full_state_dict=True)