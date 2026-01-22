import numpy as np
from collections import defaultdict
from ase.geometry.dimensionality.disjoint_set import DisjointSet
def merge_mutual_visits(all_visited, ranks, graph):
    """Find components with mutual visits and merge them."""
    merged = False
    common = defaultdict(list)
    for b, visited in all_visited.items():
        for offset in visited:
            for a in common[offset]:
                assert ranks[a] == ranks[b]
                merged |= graph.union(a, b)
            common[offset].append(b)
    if not merged:
        return (merged, all_visited, ranks)
    merged_visits = defaultdict(set)
    merged_ranks = {}
    parents = graph.find_all()
    for k, v in all_visited.items():
        key = parents[k]
        merged_visits[key].update(v)
        merged_ranks[key] = ranks[key]
    return (merged, merged_visits, merged_ranks)