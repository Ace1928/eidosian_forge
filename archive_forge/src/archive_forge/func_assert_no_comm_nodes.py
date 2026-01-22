from typing import List
import torch
from . import config, ir, scheduler
from .dependencies import WeakDep
from .utils import tuple_sorted
def assert_no_comm_nodes(snodes: List['scheduler.BaseSchedulerNode']) -> None:
    assert not any((isinstance(snode.node, ir.CollectiveKernel) for snode in snodes))