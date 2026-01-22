import logging
import os
from collections import defaultdict, namedtuple
from functools import reduce
from itertools import chain
from math import log2
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple
from fontTools.config import OPTIONS
from fontTools.misc.intTools import bit_count, bit_indices
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables import otBase, otTables
def cluster_pairs_by_class2_coverage_custom_cost(font: TTFont, pairs: Pairs, compression: int=5) -> List[Pairs]:
    if not pairs:
        return [pairs]
    all_class1 = sorted(set((pair[0] for pair in pairs)))
    all_class2 = sorted(set((pair[1] for pair in pairs)))
    lines = [sum((1 << i if (class1, class2) in pairs else 0 for i, class2 in enumerate(all_class2))) for class1 in all_class1]
    name_to_id = font.getReverseGlyphMap()
    all_class1_data = [_getClassRanges((name_to_id[name] for name in cls)) for cls in all_class1]
    all_class2_data = [_getClassRanges((name_to_id[name] for name in cls)) for cls in all_class2]
    format1 = 0
    format2 = 0
    for pair, value in pairs.items():
        format1 |= value[0].getEffectiveFormat() if value[0] else 0
        format2 |= value[1].getEffectiveFormat() if value[1] else 0
    valueFormat1_bytes = bit_count(format1) * 2
    valueFormat2_bytes = bit_count(format2) * 2
    ctx = ClusteringContext(lines, all_class1, all_class1_data, all_class2_data, valueFormat1_bytes, valueFormat2_bytes)
    cluster_cache: Dict[int, Cluster] = {}

    def make_cluster(indices: int) -> Cluster:
        cluster = cluster_cache.get(indices, None)
        if cluster is not None:
            return cluster
        cluster = Cluster(ctx, indices)
        cluster_cache[indices] = cluster
        return cluster

    def merge(cluster: Cluster, other: Cluster) -> Cluster:
        return make_cluster(cluster.indices_bitmask | other.indices_bitmask)
    clusters = [make_cluster(1 << i) for i in range(len(lines))]
    cost_before_splitting = make_cluster((1 << len(lines)) - 1).cost
    log.debug(f'        len(clusters) = {len(clusters)}')
    while len(clusters) > 1:
        lowest_cost_change = None
        best_cluster_index = None
        best_other_index = None
        best_merged = None
        for i, cluster in enumerate(clusters):
            for j, other in enumerate(clusters[i + 1:]):
                merged = merge(cluster, other)
                cost_change = merged.cost - cluster.cost - other.cost
                if lowest_cost_change is None or cost_change < lowest_cost_change:
                    lowest_cost_change = cost_change
                    best_cluster_index = i
                    best_other_index = i + 1 + j
                    best_merged = merged
        assert lowest_cost_change is not None
        assert best_cluster_index is not None
        assert best_other_index is not None
        assert best_merged is not None
        if lowest_cost_change > 0:
            cost_after_splitting = sum((c.cost for c in clusters))
            size_reduction = 1 - cost_after_splitting / cost_before_splitting
            max_new_subtables = -log2(1 - size_reduction) * compression
            log.debug(f'            len(clusters) = {len(clusters):3d}    size_reduction={size_reduction:5.2f}    max_new_subtables={max_new_subtables}')
            if compression == 9:
                max_new_subtables = len(clusters)
            if len(clusters) <= max_new_subtables + 1:
                break
        del clusters[best_other_index]
        clusters[best_cluster_index] = best_merged
    pairs_by_class1: Dict[Tuple[str, ...], Pairs] = defaultdict(dict)
    for pair, values in pairs.items():
        pairs_by_class1[pair[0]][pair] = values
    pairs_groups: List[Pairs] = []
    for cluster in clusters:
        pairs_group: Pairs = dict()
        for i in cluster.indices:
            class1 = all_class1[i]
            pairs_group.update(pairs_by_class1[class1])
        pairs_groups.append(pairs_group)
    return pairs_groups