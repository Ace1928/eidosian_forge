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
def test_invalid_devices(self):
    if self.rank != 0:
        return
    dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
    with self.assertRaisesRegex(RuntimeError, 'Expected one of .+ device type at start of device string'):
        [m.forward() for m in self._create_remote_module_iter(f'{dst_worker_name}/foo', modes=[ModuleCreationMode.MODULE_CTOR])]
    with self.assertRaisesRegex(RuntimeError, 'CUDA error: invalid device ordinal'):
        [m.forward() for m in self._create_remote_module_iter(f'{dst_worker_name}/cuda:100', modes=[ModuleCreationMode.MODULE_CTOR])]
    with self.assertRaisesRegex(RuntimeError, "Invalid device string: 'cpu2'"):
        [m.forward() for m in self._create_remote_module_iter(f'{dst_worker_name}/cpu2', modes=[ModuleCreationMode.MODULE_CTOR])]
    with self.assertRaisesRegex(RuntimeError, 'Device string must not be empty'):
        [m.forward() for m in self._create_remote_module_iter(f'{dst_worker_name}/', modes=[ModuleCreationMode.MODULE_CTOR])]
    with self.assertRaisesRegex(ValueError, "Could not parse remote_device: worker1/cuda:0/cuda:1. The valid format is '<workername>/<device>'"):
        [m.forward() for m in self._create_remote_module_iter(f'{dst_worker_name}/cuda:0/cuda:1', modes=[ModuleCreationMode.MODULE_CTOR])]
    with self.assertRaisesRegex(ValueError, "Could not parse remote_device: /. The valid format is '<workername>/<device>'"):
        [m.forward() for m in self._create_remote_module_iter('/', modes=[ModuleCreationMode.MODULE_CTOR])]
    with self.assertRaisesRegex(ValueError, "Could not parse remote_device: /cuda:0. The valid format is '<workername>/<device>'"):
        [m.forward() for m in self._create_remote_module_iter('/cuda:0', modes=[ModuleCreationMode.MODULE_CTOR])]