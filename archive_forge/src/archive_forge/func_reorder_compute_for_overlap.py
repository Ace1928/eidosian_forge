from typing import List
import torch
from . import config, ir, scheduler
from .dependencies import WeakDep
from .utils import tuple_sorted
def reorder_compute_for_overlap(snodes: List['scheduler.BaseSchedulerNode']) -> List['scheduler.BaseSchedulerNode']:
    """
    Decides a global ordering of all compute and communication nodes,
    assuming that we already have a global ordering of communication nodes.

    Overall scheduling procedure is:
        Step 1: Given that we've currently scheduled comm N, we now schedule all compute nodes
            that are required for comm N + 1 but do not depend on comm N, to run at the same time with comm N.
        Step 2: If all those compute nodes are sufficient to overlap comm N, we're done.
            Otherwise, we now need to look elsewhere to find compute that overlaps with comm N.
            We prioritize compute nodes that are needed sooner.
        Step 3: We schedule the compute nodes dependent on comm N and required for comm N + 1.
        Step 4: We schedule comm N + 1.
        Repeat this for subsequent comm nodes.
    """
    final_order = []
    comm_nodes = []
    for snode in snodes:
        if isinstance(snode.node, ir.CollectiveKernel):
            comm_nodes.append(snode)
    if len(comm_nodes) == 0:
        return snodes
    comm_ancestors = {node: get_ancestors(node) for node in comm_nodes}
    comm_descendants = {node: get_descendants(node) for node in comm_nodes}
    indeg = {k: 0 for k in snodes}
    for snode in snodes:
        for user in snode.node_users:
            if user in indeg:
                indeg[user] += 1
    ready_to_schedule_nodes = {node for node in snodes if indeg[node] == 0}
    unscheduled_nodes = set()
    unscheduled_nodes = set(snodes)

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
    assert len(comm_nodes) > 0
    schedule_nodes(list(comm_ancestors[comm_nodes[0]]) + [comm_nodes[0]])
    rolled_over_compute_cost = 0
    for idx in range(1, len(comm_ancestors)):
        needed_by_next_comm_and_ready_compute_nodes = unscheduled_nodes & comm_ancestors[comm_nodes[idx]] - comm_descendants[comm_nodes[idx - 1]]
        assert_no_comm_nodes(needed_by_next_comm_and_ready_compute_nodes)
        total_compute_runtime_cost = rolled_over_compute_cost + sum([estimate_op_runtime(node) for node in needed_by_next_comm_and_ready_compute_nodes])
        prev_comm_runtime_cost = estimate_op_runtime(comm_nodes[idx - 1])
        schedule_nodes(tuple_sorted(needed_by_next_comm_and_ready_compute_nodes))
        step1_runtime_cost = total_compute_runtime_cost
        if step1_runtime_cost >= prev_comm_runtime_cost:
            pass
        else:
            ready_to_schedule_compute_nodes = tuple_sorted(ready_to_schedule_nodes - comm_descendants[comm_nodes[idx - 1]])
            assert_no_comm_nodes(ready_to_schedule_compute_nodes)

            def earliest_comm_descendant(node):
                for idx in range(len(comm_nodes)):
                    if node in comm_ancestors[comm_nodes[idx]]:
                        return idx
                return len(comm_nodes)
            ready_to_schedule_compute_nodes = sorted(ready_to_schedule_compute_nodes, key=earliest_comm_descendant)
            for snode in ready_to_schedule_compute_nodes:
                if total_compute_runtime_cost >= prev_comm_runtime_cost:
                    break
                compute_runtime_cost = estimate_op_runtime(snode)
                if prev_comm_runtime_cost - total_compute_runtime_cost <= compute_runtime_cost / 2:
                    continue
                schedule_node(snode)
                total_compute_runtime_cost += compute_runtime_cost
        rollable_compute_cost = total_compute_runtime_cost - step1_runtime_cost
        needed_by_next_comm_nodes = unscheduled_nodes & comm_ancestors[comm_nodes[idx]]
        schedule_nodes(list(needed_by_next_comm_nodes))
        schedule_nodes([comm_nodes[idx]])
        is_prev_comm_blocking_next_comm = len(needed_by_next_comm_nodes) > 0
        if is_prev_comm_blocking_next_comm:
            rolled_over_compute_cost = 0
        else:
            rolled_over_compute_cost = rollable_compute_cost
    schedule_nodes(unscheduled_nodes)
    return final_order