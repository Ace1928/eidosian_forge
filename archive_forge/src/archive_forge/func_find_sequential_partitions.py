import itertools
from typing import Any, List, OrderedDict, Set, Optional, Callable
import operator
from torch.fx import Node
import torch
from torch.fx.passes.utils.source_matcher_utils import (
def find_sequential_partitions(gm: torch.fx.GraphModule, partition_types: List[Any], include_functional_equivalent=True, filter_fn: Optional[Callable[[Node], bool]]=None):
    if not _valid_type_sequence(partition_types):
        raise ValueError(f'Invalid partition types: {partition_types}. Each type in the sequence must be unique')
    typed_partitions: OrderedDict[Any, List[SourcePartition]] = OrderedDict()
    for partition_type in partition_types:
        types_to_match = _get_matching_types(partition_type)
        partitions = get_source_partitions(gm.graph, types_to_match, filter_fn)
        typed_partitions[partition_type] = list(itertools.chain(*partitions.values()))
    typed_partitions_list = list(typed_partitions.values())
    fusion_candidates = itertools.product(*typed_partitions_list)
    fused_partitions = []
    for candidate in fusion_candidates:
        if _partitions_sequential(candidate):
            fused_partitions.append(candidate)
    return fused_partitions