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
def deduce_node_dtype_by_inputs(self, node: torch.fx.Node):
    inputs = node.all_input_nodes
    input_nodes = [n for n in inputs if isinstance(n, torch.fx.Node) and n.op != 'placeholder']
    if len(input_nodes) == 0:
        return None
    all_input_nodes_propogated = all((OptimizationContext.key in n.meta and n.meta[OptimizationContext.key].dtype is not None for n in input_nodes))
    if not all_input_nodes_propogated:
        return None
    return functools.reduce(torch.promote_types, [n.meta[OptimizationContext.key].dtype for n in input_nodes])