from collections import deque
from contextlib import contextmanager
import threading
from typing import (
import torch
from torch import Tensor
import torch.autograd
from .dependency import fork, join
from .microbatch import Batch
from .phony import get_phony
@contextmanager
def restore_rng_states(device: torch.device, rng_states: Deque[RNGStates]) -> Generator[None, None, None]:
    """:
    Restore the random number generator state.

    meth:`Recompute.backward` restores the random number generator states
    captured by :func:`save_rng_states` within its context.

    .. seealso:: :ref:`Referential Transparency`

    """
    cpu_rng_state, gpu_rng_state = rng_states.pop()
    gpu_devices: List[torch.device] = []
    if device.type == 'cuda':
        gpu_devices.append(device)
    with torch.random.fork_rng(gpu_devices):
        torch.set_rng_state(cpu_rng_state)
        if gpu_rng_state is not None:
            torch.cuda.set_rng_state(gpu_rng_state, device)
        yield