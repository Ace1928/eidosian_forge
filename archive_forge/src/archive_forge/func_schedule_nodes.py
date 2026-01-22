from typing import List
import torch
from . import config, ir, scheduler
from .dependencies import WeakDep
from .utils import tuple_sorted
def schedule_nodes(snodes):
    """
        Schedules all nodes in `snodes` in an arbitrary topologically valid order.
        """
    all_nodes = set(snodes)
    assert all((node in unscheduled_nodes for node in all_nodes))
    while len(all_nodes) > 0:
        progress = False
        for node in tuple_sorted(all_nodes):
            if node in ready_to_schedule_nodes:
                schedule_node(node)
                all_nodes.remove(node)
                progress = True
        if not progress:
            raise Exception('Unable to find a free node (indeg == 0). This is an impossible state to reach. Please report a bug to PyTorch.')