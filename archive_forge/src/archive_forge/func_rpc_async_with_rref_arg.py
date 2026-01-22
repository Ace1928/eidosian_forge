from typing import Dict, Tuple
import torch
import torch.distributed.rpc as rpc
from torch import Tensor
from torch.distributed.rpc import RRef
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
@torch.jit.script
def rpc_async_with_rref_arg(dst_worker_name: str, args: Tuple[RRef[Tensor]]) -> Tensor:
    fut = rpc.rpc_async(dst_worker_name, rref_to_here, args)
    ret = fut.wait()
    return ret