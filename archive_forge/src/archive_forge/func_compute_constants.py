import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def compute_constants(lz, variables):
    """Fold constant arrays - everything not dependent on ``variables`` -
    into the graph.

    Parameters
    ----------
    lz : pytree of LazyArray
        The output node(s) of the computational graph.
    variables : pytree of LazyArray
        Nodes that should be treated as variable. I.e. any descendants will
        not be folded into the graph.
    """
    variables = set(tree_iter(variables, is_lazy_array))
    for node in ascend(lz):
        if not any((c in variables for c in node._deps)):
            node._materialize()
        else:
            variables.add(node)