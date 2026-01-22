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
@staticmethod
def perform_backward_for_invocation(transport: Transport, message: PipeMessage, activations: Activations, invocation: Invocation) -> None:
    """Perform the backward pass by looking up the appropriate `Batch` and
        then calling `backward` on the tensor"""
    recvd_grads = transport.recv_message_tensors(message)
    batch: Batch = activations[invocation.this.index][invocation.order][message.args.microbatch_index]
    batch.tensor.grad_fn.grad_from_pipeline = tuple(recvd_grads.tensors)
    batch.tensor.backward(retain_graph=True)