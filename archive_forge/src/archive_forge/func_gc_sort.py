from collections import deque
from . import errors, revision
def gc_sort(self):
    """Return a reverse topological ordering which is 'stable'.

        There are a few constraints:
          1) Reverse topological (all children before all parents)
          2) Grouped by prefix
          3) 'stable' sorting, so that we get the same result, independent of
             machine, or extra data.
        To do this, we use the same basic algorithm as topo_sort, but when we
        aren't sure what node to access next, we sort them lexicographically.
        """
    tips = self._find_tips()
    prefix_tips = {}
    for node in tips:
        if node.key.__class__ is str or len(node.key) == 1:
            prefix = ''
        else:
            prefix = node.key[0]
        prefix_tips.setdefault(prefix, []).append(node)
    num_seen_children = dict.fromkeys(self._nodes, 0)
    result = []
    for prefix in sorted(prefix_tips):
        pending = sorted(prefix_tips[prefix], key=lambda n: n.key, reverse=True)
        while pending:
            node = pending.pop()
            if node.parent_keys is None:
                continue
            result.append(node.key)
            for parent_key in sorted(node.parent_keys, reverse=True):
                parent_node = self._nodes[parent_key]
                seen_children = num_seen_children[parent_key] + 1
                if seen_children == len(parent_node.child_keys):
                    pending.append(parent_node)
                    del num_seen_children[parent_key]
                else:
                    num_seen_children[parent_key] = seen_children
    return result