from collections import deque
import contextlib
import functools
from itertools import chain
import logging
from typing import Any, Callable, Deque, Dict, Generator, List, Optional, Union
import torch
from torch import nn
from torch.autograd import Variable
import torch.autograd.profiler as profiler
import torch.distributed as dist
from fairscale.internal.params import Workhandle, get_global_rank
from fairscale.nn.misc import GradBucket
from fairscale.optim import OSS
@torch.no_grad()
def sync_buffers(self, blocking: bool=False) -> None:
    """
        Sync all the param buffers in between ranks (including for instance batch norm statistics).

        Arguments:
            blocking (bool): wait for the operation to conclude.
        """
    with profiler.record_function('fairscale::sdp::sync_buffers'):
        work_handles = []
        for buffer in self.module.buffers(recurse=True):
            work_handles.append(dist.broadcast(buffer.data, self._reference_global_rank, self._process_group, async_op=True))
        if blocking and work_handles:
            if self._backend != dist.Backend.NCCL:
                _ = list(filter(lambda x: x.wait(), work_handles))
            else:
                work_handles[-1].wait()