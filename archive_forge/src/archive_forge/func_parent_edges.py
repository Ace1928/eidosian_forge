from functools import reduce
def parent_edges(self, child):
    """Return a list of (parent, label) pairs for child."""
    if child not in self._adjacency_list:
        raise ValueError('Unknown <child> node: ' + str(child))
    parents = []
    for parent, children in self._adjacency_list.items():
        for x in children:
            if x == child:
                parents.append((parent, self._edge_map[parent, child]))
    return sorted(parents)