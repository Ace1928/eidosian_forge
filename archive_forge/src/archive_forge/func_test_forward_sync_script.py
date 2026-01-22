import enum
from typing import Tuple
import torch
import torch.distributed.rpc as rpc
import torch.testing._internal.dist_utils as dist_utils
from torch import Tensor, nn
from torch._jit_internal import Future
from torch.distributed.nn import RemoteModule
from torch.distributed.nn.api.remote_module import _REMOTE_MODULE_PICKLED_ATTRIBUTES
from torch.distributed.nn.api.remote_module import _RemoteModule
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import TemporaryFileName
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
@dist_utils.dist_init
def test_forward_sync_script(self):
    if self.rank != 0:
        return
    dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
    scripted_remote_module = next(self._create_remote_module_iter(dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR_WITH_INTERFACE]))

    @torch.jit.script
    def run_forward(scripted_remote_module: MyModuleInterface):
        ret = scripted_remote_module.forward(torch.ones(1), 2, '3')
        return ret
    ret = run_forward(scripted_remote_module)
    self.assertEqual(ret, ('3', 2, torch.ones(1)))