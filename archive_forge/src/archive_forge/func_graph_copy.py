import collections
from collections import defaultdict
from .node import Node, Argument, Target, map_arg, _type_repr, _get_qualified_name
import torch.utils._pytree as pytree
from . import _pytree as fx_pytree
from ._compatibility import compatibility
import contextlib
from typing import TYPE_CHECKING, Callable, Any, List, Dict, NamedTuple, Optional, Tuple, Set, FrozenSet, Type
from dataclasses import dataclass
from contextlib import contextmanager
import copy
import enum
import torch
import keyword
import re
import builtins
import math
import warnings
import inspect
@compatibility(is_backward_compatible=True)
def graph_copy(self, g: 'Graph', val_map: Dict[Node, Node], return_output_node=False) -> 'Optional[Argument]':
    """
        Copy all nodes from a given graph into ``self``.

        Args:

            g (Graph): The source graph from which to copy Nodes.

            val_map (Dict[Node, Node]): a dictionary that will be populated with a mapping
                from nodes in ``g`` to nodes in ``self``. Note that ``val_map`` can be passed
                in with values in it already to override copying of certain values.

        Returns:

            The value in ``self`` that is now equivalent to the output value in ``g``,
            if ``g`` had an ``output`` node. ``None`` otherwise.
        """
    for node in g.nodes:
        if node in val_map:
            continue
        if node.op == 'output':
            rv = map_arg(node.args[0], lambda n: val_map[n])
            return rv if not return_output_node else (rv, node)
        val_map[node] = self.node_copy(node, lambda n: val_map[n])
    return None