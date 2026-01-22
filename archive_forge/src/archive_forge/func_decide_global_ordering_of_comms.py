from typing import List
import torch
from . import config, ir, scheduler
from .dependencies import WeakDep
from .utils import tuple_sorted
def decide_global_ordering_of_comms(nodes: List['scheduler.BaseSchedulerNode']):
    """
    Decide global ordering of comms, by just enforcing the ordering that's in the input graph
    (might not be the same ordering as the eager mode program).
    TODO: Come up with a better approach
    """
    comm_nodes = [n for n in nodes if isinstance(n.node, ir.CollectiveKernel)]
    for i in range(1, len(comm_nodes)):
        comm_nodes[i].add_fake_dep(WeakDep(comm_nodes[i - 1].get_name()))