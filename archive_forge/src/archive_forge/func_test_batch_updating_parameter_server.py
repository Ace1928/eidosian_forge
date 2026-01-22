import threading
from datetime import datetime
from time import perf_counter
import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
from torch import optim
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import RpcAgentTestFixture
@dist_init(setup_rpc=False)
def test_batch_updating_parameter_server(self):
    if self.rank != 0:
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=self.rpc_backend_options)
    else:
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=self.rpc_backend_options)
        run_ps([f'{worker_name(r)}' for r in range(1, self.world_size)])
    rpc.shutdown()