import collections
import logging
import operator
from typing import Any, DefaultDict, Deque, Dict, Iterator, List, Optional, Set, Tuple
import torch
from torch._dynamo.utils import counters
from torch._utils_internal import print_graph
from .. import config
from ..pattern_matcher import (
def generate_fusion_from_config(config_options: Dict[str, Any], pre_grad=True):
    fusions: List[GroupBatchFusionBase] = []
    for name, options in config_options.items():
        fusion_cls = PRE_GRAD_FUSIONS[name] if pre_grad else POST_GRAD_FUSIONS[name]
        _options = graph_search_options.copy()
        _options.update(options)
        fusions.append(fusion_cls(graph_search_options=_options))
    return fusions