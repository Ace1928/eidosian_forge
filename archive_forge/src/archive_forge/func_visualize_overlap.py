from typing import List
import torch
from . import config, ir, scheduler
from .dependencies import WeakDep
from .utils import tuple_sorted
def visualize_overlap(order):
    total_est_runtime: float = 0.0
    cur_comm_node = None
    for snode in order:
        if cur_comm_node is None:
            if isinstance(snode.node, ir.CollectiveKernel):
                total_est_runtime += estimate_op_runtime(snode)
                cur_comm_node = snode.node
            elif isinstance(snode.node, ir.Wait):
                raise Exception('Wait is not expected when there is no collective running')
            else:
                total_est_runtime += estimate_op_runtime(snode)
            overlap_log.debug(f'{node_summary(snode)}')
        elif isinstance(snode.node, ir.CollectiveKernel):
            raise Exception('Found two collectives running at the same time. `visualize_overlap` needs to be updated to handle this case')
        elif isinstance(snode.node, ir.Wait):
            overlap_log.debug(f'{node_summary(snode)}')
            cur_comm_node = None
        else:
            overlap_log.debug(f'| {node_summary(snode)}')
    overlap_log.debug(f'Est. runtime (ms): {total_est_runtime / 1000 / 1000}')