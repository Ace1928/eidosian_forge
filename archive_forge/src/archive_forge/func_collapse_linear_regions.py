import time
from . import debug, errors, osutils, revision, trace
def collapse_linear_regions(parent_map):
    """Collapse regions of the graph that are 'linear'.

    For example::

      A:[B], B:[C]

    can be collapsed by removing B and getting::

      A:[C]

    :param parent_map: A dictionary mapping children to their parents
    :return: Another dictionary with 'linear' chains collapsed
    """
    children = {}
    for child, parents in parent_map.items():
        children.setdefault(child, [])
        for p in parents:
            children.setdefault(p, []).append(child)
    removed = set()
    result = dict(parent_map)
    for node in parent_map:
        parents = result[node]
        if len(parents) == 1:
            parent_children = children[parents[0]]
            if len(parent_children) != 1:
                continue
            node_children = children[node]
            if len(node_children) != 1:
                continue
            child_parents = result.get(node_children[0], None)
            if len(child_parents) != 1:
                continue
            result[node_children[0]] = parents
            children[parents[0]] = node_children
            del result[node]
            del children[node]
            removed.add(node)
    return result