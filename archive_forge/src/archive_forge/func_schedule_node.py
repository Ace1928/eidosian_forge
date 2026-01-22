from typing import List
import torch
from . import config, ir, scheduler
from .dependencies import WeakDep
from .utils import tuple_sorted
def schedule_node(snode):
    """
        Schedule a single node.
        """
    assert snode in unscheduled_nodes
    assert snode in ready_to_schedule_nodes
    ready_to_schedule_nodes.remove(snode)
    unscheduled_nodes.remove(snode)
    final_order.append(snode)
    for user in tuple_sorted(snode.node_users):
        if user in indeg:
            indeg[user] -= 1
            if indeg[user] == 0:
                ready_to_schedule_nodes.add(user)