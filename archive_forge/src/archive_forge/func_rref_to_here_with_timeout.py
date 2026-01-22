from typing import Dict, Tuple
import torch
import torch.distributed.rpc as rpc
from torch import Tensor
from torch.distributed.rpc import RRef
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
@torch.jit.script
def rref_to_here_with_timeout(rref_var: RRef[Tensor], timeout: float) -> Tensor:
    return rref_var.to_here(timeout)