import torch.distributed.rpc as rpc
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.testing._internal.common_distributed import (
@property
def rpc_backend_options(self):
    return rpc.backend_registry.construct_rpc_backend_options(self.rpc_backend, init_method=self.init_method, _transports=tp_transports())