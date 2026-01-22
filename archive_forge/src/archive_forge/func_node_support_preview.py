import argparse
import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple, Sequence, Iterable, Any, List, Dict, Optional, Tuple
import logging
import torch
from torch.fx.passes.graph_manipulation import get_size_of_node
from torch.fx.node import map_arg
from torch.fx._compatibility import compatibility
from .operator_support import (
from .graph_drawer import FxGraphDrawer
from .shape_prop import ShapeProp
from .split_utils import split_by_tags
from .tools_common import (
def node_support_preview(self, dump_graph: bool=False):
    submodules = dict(self.module.named_modules())
    supported_nodes: NodeList = []
    supported_node_types = defaultdict(set)
    unsupported_node_types = defaultdict(set)

    def get_dtype(arg):
        tensor_meta = arg.meta.get('tensor_meta')
        return getattr(tensor_meta, 'dtype', None)
    for node in self.module.graph.nodes:
        if node.op not in CALLABLE_NODE_OPS:
            continue
        target = get_node_target(submodules, node)
        arg_dtypes = [get_dtype(arg) if isinstance(arg, torch.fx.Node) else None for arg in node.args]
        last_index = len(arg_dtypes) - next((i for i, dtype in enumerate(reversed(arg_dtypes)) if dtype is not None), len(arg_dtypes))
        arg_dtypes_tuple = tuple(arg_dtypes[:last_index])
        kwarg_dtypes_tuple = tuple(((k, get_dtype(arg)) for k, arg in node.kwargs.items() if isinstance(arg, torch.fx.Node)))
        if self.operator_support.is_node_supported(submodules, node):
            supported_nodes.append(node)
            supported_node_types[target].add((arg_dtypes_tuple, kwarg_dtypes_tuple))
        else:
            unsupported_node_types[target].add((arg_dtypes_tuple, kwarg_dtypes_tuple))
    if dump_graph:
        self._draw_graph_based_on_node_support(self.module, supported_nodes)
    reports = '\nSupported node types in the model:\n'
    for t, dtypes in supported_node_types.items():
        for arg_dtypes_tuple, kwarg_dtypes_tuple in dtypes:
            reports += f'{t}: ({arg_dtypes_tuple}, {dict(kwarg_dtypes_tuple)})\n'
    reports += '\nUnsupported node types in the model:\n'
    for t, dtypes in unsupported_node_types.items():
        for arg_dtypes_tuple, kwarg_dtypes_tuple in dtypes:
            reports += f'{t}: ({arg_dtypes_tuple}, {dict(kwarg_dtypes_tuple)})\n'
    print(reports)
    return reports