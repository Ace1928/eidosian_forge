import copy
import torch
from torch.distributed._shard.sharded_tensor import (
from ._common import (
from torch.distributed._shard.common_op_utils import _register_default_op
def sharded_detach(args, kwargs, pg):
    self_st = args[0]
    detached_local_shards = [Shard(local_shard.tensor.detach(), metadata=copy.deepcopy(local_shard.metadata)) for local_shard in self_st.local_shards()]
    new_metadata = copy.deepcopy(self_st.metadata())
    new_metadata.tensor_properties.requires_grad = False
    return (detached_local_shards, new_metadata)