import collections
import io
import sys
import types
from typing import (
import torch
import torch.distributed.rpc as rpc
from torch import Tensor, device, dtype, nn
from torch.distributed.nn.jit import instantiator
from torch.distributed import _remote_device
from torch.distributed.rpc.internal import _internal_rpc_pickler
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.utils.hooks import RemovableHandle
def remote_parameters(self, recurse: bool=True) -> List[rpc.RRef[Parameter]]:
    """
        Return a list of :class:`~torch.distributed.rpc.RRef` pointing to the remote module's parameters.

        This can typically be used in conjunction
        with :class:`~torch.distributed.optim.DistributedOptimizer`.

        Args:
            recurse (bool): if True, then returns parameters of the remote
                module and all submodules of the remote module. Otherwise,
                returns only parameters that are direct members of the
                remote module.

        Returns:
            A list of :class:`~torch.distributed.rpc.RRef` (``List[RRef[nn.Parameter]]``)
            to remote module's parameters.
        """
    return rpc.rpc_sync(self.on, _param_rrefs, args=(self.module_rref, recurse))