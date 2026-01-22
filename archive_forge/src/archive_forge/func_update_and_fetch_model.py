import threading
from datetime import datetime
from time import perf_counter
import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
from torch import optim
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import RpcAgentTestFixture
@staticmethod
@rpc.functions.async_execution
def update_and_fetch_model(ps_rref, grads):
    self = ps_rref.local_value()
    for p, g in zip(self.model.parameters(), grads):
        if p.grad is None:
            p.grad = g
        else:
            p.grad += g
    with self.lock:
        timed_log(f'PS got {self.curr_update_size}/{self.batch_update_size} updates')
        self.curr_update_size += 1
        fut = self.future_model
        if self.curr_update_size >= self.batch_update_size:
            for p in self.model.parameters():
                p.grad /= self.batch_update_size
            self.curr_update_size = 0
            self.optimizer.step()
            self.optimizer.zero_grad()
            fut.set_result(self.model)
            timed_log('PS updated model')
            self.future_model = torch.futures.Future()
    return fut