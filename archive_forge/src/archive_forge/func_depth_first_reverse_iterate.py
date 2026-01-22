import collections
import enum
from taskflow.engines.action_engine import compiler as co
def depth_first_reverse_iterate(node, start_from_idx=-1):
    """Iterates connected (in reverse) **tree** nodes (from starting node).

    Jumps through nodes with ``noop`` attribute (does not yield them back).
    """
    if start_from_idx == -1:
        children_iter = node.reverse_iter()
    else:
        children_iter = reversed(node[0:start_from_idx])
    for child in children_iter:
        if child.metadata.get('noop'):
            for grand_child in child.dfs_iter(right_to_left=False):
                if grand_child.metadata['kind'] in co.ATOMS:
                    yield grand_child.item
        else:
            yield child.item