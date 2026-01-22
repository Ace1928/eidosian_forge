import contextlib
import enum
import logging
import os
import threading
from typing import NamedTuple
import torch
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.nn as nn
from torch.distributed import rpc
from torch.distributed.nn import RemoteModule
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import (
from torch.testing._internal.dist_utils import INIT_METHOD_TEMPLATE, dist_init
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
@requires_gloo()
@dist_init
def test_ddp_dist_autograd_sparse_grads(self):
    torch.manual_seed(self.rank)
    dist.init_process_group(backend='gloo', init_method=INIT_METHOD_TEMPLATE.format(file_name=self.file_name), world_size=self.world_size, rank=self.rank)
    model = nn.EmbeddingBag(10, 3, sparse=True)
    ddp_model = DistributedDataParallel(model)
    input = torch.LongTensor(10).random_(0, 10)
    offsets = torch.LongTensor([0, 4])
    loss = ddp_model(input, offsets).sum()
    loss.backward()
    with dist_autograd.context() as context_id:
        loss = ddp_model(input, offsets).sum()
        dist_autograd.backward(context_id, [loss])
        grads_dict = dist_autograd.get_gradients(context_id)
        self.assertEqual(1, len(grads_dict))
        self.assertEqual(model.weight.grad, grads_dict[model.weight])