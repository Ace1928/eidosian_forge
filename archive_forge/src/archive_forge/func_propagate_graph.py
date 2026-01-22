import contextlib
import dataclasses
import functools
import itertools
import logging
import operator
import re
from collections import namedtuple
from itertools import chain
from typing import (
import sympy
from sympy.printing.printer import Printer
import torch
import torch.fx
from torch.utils._sympy.value_ranges import ValueRanges
from .. import config, metrics
from ..utils import (
from ..virtualized import ops, OpsValue, V
def propagate_graph(self, graph: torch.fx.Graph):
    assert graph.nodes
    graph_dtype = None
    for node in graph.nodes:
        if OptimizationContext.key in node.meta:
            opt_ctx = node.meta[OptimizationContext.key]
        else:
            opt_ctx = OptimizationContext()
        opt_ctx.dtype = self.deduce_node_dtype(node)
        node.meta[OptimizationContext.key] = opt_ctx
        if node.target == 'output':
            graph_dtype = opt_ctx.dtype
    return graph_dtype