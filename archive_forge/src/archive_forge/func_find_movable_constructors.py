import functools
import itertools
import logging
import operator
from collections import Counter, defaultdict, namedtuple
from typing import Any, Dict, List, Optional, Set, Union
from sympy import Expr
import torch
import torch._inductor as inductor
import torch.utils._pytree as pytree
from torch import fx
from torch._decomp import register_decomposition
from torch._higher_order_ops.triton_kernel_wrap import triton_kernel_wrapper_functional
from torch._prims_common import is_boolean_dtype, is_expandable_to, is_integer_dtype
from torch._utils_internal import print_graph
from torch.fx.experimental.symbolic_shapes import definitely_true, sym_eq
from torch.fx.immutable_collections import immutable_dict
from .. import config, inductor_prims, ir, pattern_matcher
from ..fx_utils import FakeTensorUpdater, get_fake_args_kwargs, get_node_storage
from ..lowering import (
from ..pattern_matcher import (
from ..utils import decode_device, is_pointwise_use
from ..virtualized import V
from .group_batch_fusion import group_batch_fusion_passes
def find_movable_constructors(self, graph: fx.Graph, constructors: List[fx.Node]) -> Set[fx.Node]:
    """
        Starting from the cpu constructors, iterate through the graph and test that all of their
        downstream uses can safely be moved to cpu.
        """
    cpu_indeg: Dict[fx.Node, int] = self.get_cpu_indeg_count(graph)
    cannot_move_to_cuda: Set[fx.Node] = set()
    constructor_dependencies: Dict[fx.Node, Set[fx.Node]] = defaultdict(set)
    equal_constructor_sets: Dict[fx.Node, Set[fx.Node]] = {c: {c} for c in constructors}

    def make_dependencies_equivalent(set1: Set[fx.Node], set2: Set[fx.Node]) -> Set[fx.Node]:
        set1.update(set2)
        for obj in set1:
            equal_constructor_sets[obj] = set1
        return set1
    queue: List[fx.Node] = list(constructors)
    for c in queue:
        constructor_dependencies[c].add(c)
    while queue:
        node = queue.pop()
        dependencies = constructor_dependencies[node]
        for user in node.users:
            if self.cannot_be_moved(user):
                cannot_move_to_cuda.update(dependencies)
                break
            node_device = self.get_node_device(user)
            if self.allow_cpu_device(user) and node_device and (node_device.type == self.target):
                del cpu_indeg[user]
            else:
                cpu_indeg[user] -= 1
                if cpu_indeg[user] == 0:
                    del cpu_indeg[user]
                    queue.append(user)
            unioned_set = make_dependencies_equivalent(dependencies, constructor_dependencies[user])
            constructor_dependencies[user] = unioned_set
    for node in cpu_indeg:
        if constructor_dependencies[node]:
            cannot_move_to_cuda.update(constructor_dependencies[node])
    all_cannot_move_to_cuda = cannot_move_to_cuda.copy()
    for constructor in cannot_move_to_cuda:
        all_cannot_move_to_cuda.update(equal_constructor_sets[constructor])
    return set(constructors) - all_cannot_move_to_cuda