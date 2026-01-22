from . import errors
from . import graph as _mod_graph
from . import revision as _mod_revision
def iter_topo_order(self):
    """Yield the nodes of the graph in a topological order.

        After finishing iteration the sorter is empty and you cannot continue
        iteration.
        """
    node_name_stack = self._node_name_stack
    node_merge_depth_stack = self._node_merge_depth_stack
    pending_parents_stack = self._pending_parents_stack
    left_subtree_pushed_stack = self._left_subtree_pushed_stack
    completed_node_names = self._completed_node_names
    scheduled_nodes = self._scheduled_nodes
    graph_pop = self._graph.pop

    def push_node(node_name, merge_depth, parents, node_name_stack_append=node_name_stack.append, node_merge_depth_stack_append=node_merge_depth_stack.append, left_subtree_pushed_stack_append=left_subtree_pushed_stack.append, pending_parents_stack_append=pending_parents_stack.append, first_child_stack_append=self._first_child_stack.append, revnos=self._revnos):
        """Add node_name to the pending node stack.

            Names in this stack will get emitted into the output as they are popped
            off the stack.

            This inlines a lot of self._variable.append functions as local
            variables.
            """
        node_name_stack_append(node_name)
        node_merge_depth_stack_append(merge_depth)
        left_subtree_pushed_stack_append(False)
        pending_parents_stack_append(list(parents))
        parent_info = None
        if parents:
            try:
                parent_info = revnos[parents[0]]
            except KeyError:
                pass
        if parent_info is not None:
            first_child = parent_info[1]
            parent_info[1] = False
        else:
            first_child = None
        first_child_stack_append(first_child)

    def pop_node(node_name_stack_pop=node_name_stack.pop, node_merge_depth_stack_pop=node_merge_depth_stack.pop, first_child_stack_pop=self._first_child_stack.pop, left_subtree_pushed_stack_pop=left_subtree_pushed_stack.pop, pending_parents_stack_pop=pending_parents_stack.pop, original_graph=self._original_graph, revnos=self._revnos, completed_node_names_add=self._completed_node_names.add, scheduled_nodes_append=scheduled_nodes.append, revno_to_branch_count=self._revno_to_branch_count):
        """Pop the top node off the stack

            The node is appended to the sorted output.
            """
        node_name = node_name_stack_pop()
        merge_depth = node_merge_depth_stack_pop()
        first_child = first_child_stack_pop()
        left_subtree_pushed_stack_pop()
        pending_parents_stack_pop()
        parents = original_graph[node_name]
        parent_revno = None
        if parents:
            try:
                parent_revno = revnos[parents[0]][0]
            except KeyError:
                pass
        if parent_revno is not None:
            if not first_child:
                base_revno = parent_revno[0]
                branch_count = revno_to_branch_count.get(base_revno, 0)
                branch_count += 1
                revno_to_branch_count[base_revno] = branch_count
                revno = (parent_revno[0], branch_count, 1)
            else:
                revno = parent_revno[:-1] + (parent_revno[-1] + 1,)
        else:
            root_count = revno_to_branch_count.get(0, -1)
            root_count += 1
            if root_count:
                revno = (0, root_count, 1)
            else:
                revno = (1,)
            revno_to_branch_count[0] = root_count
        revnos[node_name][0] = revno
        completed_node_names_add(node_name)
        scheduled_nodes_append((node_name, merge_depth, revno))
        return node_name
    while node_name_stack:
        parents_to_visit = pending_parents_stack[-1]
        if not parents_to_visit:
            pop_node()
        else:
            while pending_parents_stack[-1]:
                if not left_subtree_pushed_stack[-1]:
                    next_node_name = pending_parents_stack[-1].pop(0)
                    is_left_subtree = True
                    left_subtree_pushed_stack[-1] = True
                else:
                    next_node_name = pending_parents_stack[-1].pop()
                    is_left_subtree = False
                if next_node_name in completed_node_names:
                    continue
                try:
                    parents = graph_pop(next_node_name)
                except KeyError:
                    if next_node_name in self._original_graph:
                        raise errors.GraphCycleError(node_name_stack)
                    else:
                        continue
                next_merge_depth = 0
                if is_left_subtree:
                    next_merge_depth = 0
                else:
                    next_merge_depth = 1
                next_merge_depth = node_merge_depth_stack[-1] + next_merge_depth
                push_node(next_node_name, next_merge_depth, parents)
                break
    sequence_number = 0
    stop_revision = self._stop_revision
    generate_revno = self._generate_revno
    original_graph = self._original_graph
    while scheduled_nodes:
        node_name, merge_depth, revno = scheduled_nodes.pop()
        if node_name == stop_revision:
            return
        if not len(scheduled_nodes):
            end_of_merge = True
        elif scheduled_nodes[-1][1] < merge_depth:
            end_of_merge = True
        elif scheduled_nodes[-1][1] == merge_depth and scheduled_nodes[-1][0] not in original_graph[node_name]:
            end_of_merge = True
        else:
            end_of_merge = False
        if generate_revno:
            yield (sequence_number, node_name, merge_depth, revno, end_of_merge)
        else:
            yield (sequence_number, node_name, merge_depth, end_of_merge)
        sequence_number += 1