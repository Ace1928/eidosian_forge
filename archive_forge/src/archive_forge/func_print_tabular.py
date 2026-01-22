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
def print_tabular(self):
    """
        Prints the intermediate representation of the graph in tabular
        format. Note that this API requires the ``tabulate`` module to be
        installed.
        """
    try:
        from tabulate import tabulate
    except ImportError:
        print('`print_tabular` relies on the library `tabulate`, which could not be found on this machine. Run `pip install tabulate` to install the library.')
        raise
    node_specs = [[n.op, n.name, n.target, n.args, n.kwargs] for n in self.nodes]
    print(tabulate(node_specs, headers=['opcode', 'name', 'target', 'args', 'kwargs']))