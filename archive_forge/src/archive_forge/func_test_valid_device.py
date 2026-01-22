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
@skip_if_lt_x_gpu(1)
@dist_utils.dist_init
def test_valid_device(self):
    if self.rank != 0:
        return
    dst_rank = (self.rank + 1) % self.world_size
    dst_worker_name = dist_utils.worker_name(dst_rank)
    for remote_module in self._create_remote_module_iter(f'{dst_worker_name}/cuda:0', modes=[ModuleCreationMode.MODULE_CTOR]):
        device = rpc.rpc_sync(dst_worker_name, remote_device, (remote_module.module_rref,))
        self.assertEqual(device.type, 'cuda')
        self.assertEqual(device.index, 0)
    for remote_module in self._create_remote_module_iter(f'rank:{dst_rank}/cuda:0', modes=[ModuleCreationMode.MODULE_CTOR]):
        device = rpc.rpc_sync(dst_worker_name, remote_device, (remote_module.module_rref,))
        self.assertEqual(device.type, 'cuda')
        self.assertEqual(device.index, 0)