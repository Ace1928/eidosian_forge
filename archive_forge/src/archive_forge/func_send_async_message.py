from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, auto
from threading import Event
from typing import Dict, Iterable, List, Optional, Tuple
import torch
from torch import Tensor, nn
from torch.autograd.profiler import record_function
from torch.distributed import ProcessGroup
from fairscale.nn.model_parallel import get_pipeline_parallel_ranks
from .checkpoint import Checkpointing
from .messages import Transport
from .microbatch import Batch
from .skip.tracker import SkipTrackerThroughPotals, use_skip_tracker
from .types import EVENT_LOOP_QUEUE, PipeMessage, TensorOrTensors, Tensors
from .worker import Task
def send_async_message(self, dst_rank: int, result: Batch, invocation: Invocation) -> Batch:
    """Send batch to dst_rank, and use AutogradWithoutActivations to delete
        the activations since we no longer need them"""
    assert invocation.dest
    src_rank = torch.distributed.get_rank()
    body = AsyncMessageBody(AsyncMessageType.Activations, result.index, invocation.this, invocation.dest, invocation.order + 1)
    self.transport.send_message(PipeMessage(src_rank, dst_rank, queue_name=EVENT_LOOP_QUEUE, args=body, tensors=tuple([*result])), sync=True)
    phony = AutogradWithoutActivations.apply(*result)
    return Batch(phony, result.index)