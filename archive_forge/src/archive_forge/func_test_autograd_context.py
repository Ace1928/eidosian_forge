import sys
import threading
import time
from enum import Enum
import random
import torch
import torch.nn as nn
from datetime import timedelta
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.testing._internal.dist_utils
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributed.rpc import RRef
from torch.testing._internal.common_utils import IS_MACOS, skip_but_pass_in_sandcastle_if
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
@dist_init
def test_autograd_context(self):
    max_auto_increment = 281474976710655
    self.assertEqual(max_auto_increment + (self.worker_id << 48), dist_autograd._get_max_id())
    context_ids = []
    for i in range(200):
        with dist_autograd.context() as context_id:
            self.assertEqual(context_id, dist_autograd._retrieve_context(context_id)._context_id())
            self.assertEqual(self.worker_id, context_id >> 48)
            context_ids.append(context_id)
    for context_id in context_ids:
        with self.assertRaisesRegex(RuntimeError, f'Could not find autograd context with id: {context_id}'):
            dist_autograd._retrieve_context(context_id)