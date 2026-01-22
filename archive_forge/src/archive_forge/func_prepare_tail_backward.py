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
def prepare_tail_backward(batch: Batch, activations: Activations, invocations: Invocations, count_per_order: Dict[int, int], expected_gradients: int) -> None:
    if expected_gradients > 0:
        grad_fn = next((b.grad_fn for b in batch if b.requires_grad))
        assert grad_fn
        grad_fn.tail_ctx = TailBackwardContext(activations, invocations, count_per_order, expected_gradients)