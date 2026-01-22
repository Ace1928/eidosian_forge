import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
def find_anchor_nodes(self, ctx: MatchContext, searched: Set[torch.fx.Node]):
    for pattern in self.flat_args_kwargs[0]:
        if isinstance(pattern, PatternExpr):
            for other_node in pattern.find_anchor_nodes(ctx, searched):
                if not isinstance(other_node, torch.fx.Node):
                    continue
                for node in other_node.users:
                    if node not in searched:
                        if self._match_fns(node):
                            yield node
                            searched.add(node)