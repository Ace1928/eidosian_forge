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
def run_invocations_on_batch(self, batch: Batch, invocations: Invocations, order: int, skip_trackers: List[SkipTrackerThroughPotals], activations: Activations) -> Tuple[int, int]:
    """Run invocations on the batch until we hit one that receives its input
        from a different stage (i.e. another process)"""
    invocations_handled = 0
    last_order = 0
    for invocation in invocations.values():
        if invocation.order < order:
            continue
        pi = invocation.this.index
        partition = self.partitions[pi]
        if invocation.order == order:
            invocations_handled += 1
            last_order = invocation.order
            activations[pi][invocation.order][batch.index] = self.run_invocation(batch, partition, skip_trackers, invocation)
        elif invocation.source and invocation.source.stage == self.group.rank():
            invocations_handled += 1
            last_order = invocation.order
            batch = activations[invocation.source.index][invocation.order - 1][batch.index]
            activations[pi][invocation.order][batch.index] = self.run_invocation(batch, partition, skip_trackers, invocation)
            del activations[invocation.source.index][invocation.order - 1][batch.index]
        elif invocation.source and invocation.source.stage != self.group.rank():
            break
    return (invocations_handled, last_order)