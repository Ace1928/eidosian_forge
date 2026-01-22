import time
from typing import Any, Dict, List, Tuple, Union
import torch
from torch import nn
from torch.autograd.profiler import record_function
from torch.distributed import ProcessGroup
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from fairscale.nn.model_parallel import get_pipeline_parallel_ranks
from fairscale.nn.pipe.async_schedule import (
from fairscale.nn.pipe.checkpoint import Checkpointing
from fairscale.nn.pipe.messages import Transport
from fairscale.nn.pipe.microbatch import Batch
from fairscale.nn.pipe.types import (
from fairscale.nn.pipe.worker import Task
def event_loop_across_minibatches(self, lm_dataloader: DataLoader, criterion: nn.Module, optimizer: Optimizer, transform_logger_object: Any) -> None:
    activations: Dict[int, Batch] = dict()
    num_microbatch = len(lm_dataloader)
    num_activations = 0
    num_gradients = 0
    ranks = get_pipeline_parallel_ranks()
    N = len(ranks)
    cur_rank = torch.distributed.get_rank()
    n_warmup = ranks[-1] - cur_rank
    for _ in range(n_warmup):
        if self.weight_prediction:
            optimizer.update_weight_using_future_predictions(cur_rank, N, num_activations, self.chunks, forward=True)
        message = self.event_loop_trunk_forward_helper(activations)
        self.transport.send_message(message, sync=True)
        num_activations += 1
    while num_activations < num_microbatch:
        if self.weight_prediction:
            optimizer.update_weight_using_future_predictions(cur_rank, N, num_activations, self.chunks, forward=True)
        message = self.event_loop_trunk_forward_helper(activations)
        num_activations += 1
        if self.weight_prediction:
            optimizer.update_weight_using_future_predictions(cur_rank, N, num_gradients, self.chunks, forward=False)
        self.event_loop_trunk_backward_helper(activations)
        num_gradients += 1
        if self.perform_optimizer_step(optimizer, num_gradients):
            optimizer.step()
            optimizer.zero_grad()
            transform_logger_object.check_and_save_weights(num_gradients)
        self.transport.send_message(message, sync=True)
    remaining = len(activations)
    for _ in range(remaining):
        if self.weight_prediction:
            optimizer.update_weight_using_future_predictions(cur_rank, N, num_gradients, self.chunks, forward=False)
        self.event_loop_trunk_backward_helper(activations)
        num_gradients += 1
        if self.perform_optimizer_step(optimizer, num_gradients):
            optimizer.step()
            optimizer.zero_grad()
            transform_logger_object.check_and_save_weights(num_gradients)