from functools import reduce
def child_edges(self, parent):
    """Return a list of (child, label) pairs for parent."""
    if parent not in self._adjacency_list:
        raise ValueError('Unknown <parent> node: ' + str(parent))
    return [(x, self._edge_map[parent, x]) for x in sorted(self._adjacency_list[parent])]