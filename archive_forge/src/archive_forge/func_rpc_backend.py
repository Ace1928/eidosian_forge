import torch.distributed.rpc as rpc
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.testing._internal.common_distributed import (
@property
def rpc_backend(self):
    return rpc.backend_registry.BackendType['TENSORPIPE']