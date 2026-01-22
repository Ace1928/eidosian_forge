from .graph_module import GraphModule
from .graph import Graph
from .node import Node
from ._symbolic_trace import symbolic_trace
from ._compatibility import compatibility
import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Union
import torch
@compatibility(is_backward_compatible=False)
def replace_pattern_with_filters(gm: GraphModule, pattern: Union[Callable, Graph, GraphModule], replacement: Union[Callable, Graph, GraphModule], match_filters: Optional[List[Callable[['InternalMatch', Graph, Graph], bool]]]=None, ignore_literals: bool=False) -> List[ReplacedPatterns]:
    """
    See replace_pattern for documentation. This function is an overload with an additional match_filter argument.

    Args:
        ``match_filters``: A list of functions that take in
            (match: InternalMatch, original_graph: Graph, pattern_graph: Graph) and return a boolean indicating
            whether the match satisfies the condition.
            See matcher_utils.py for definition of InternalMatch.
    """
    return _replace_pattern(gm, pattern, replacement, match_filters, ignore_literals)