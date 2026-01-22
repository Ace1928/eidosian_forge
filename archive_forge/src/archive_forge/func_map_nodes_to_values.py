from .graph_module import GraphModule
from .graph import Graph
from .node import Argument, Node, Target, map_arg, map_aggregate
from .proxy import Proxy
from ._symbolic_trace import Tracer
from ._compatibility import compatibility
from . import config
import torch.fx.traceback as fx_traceback
import torch
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import inspect
from contextlib import contextmanager
from torch.hub import tqdm
@compatibility(is_backward_compatible=True)
def map_nodes_to_values(self, args: Argument, n: Node) -> Argument:
    """
        Recursively descend through ``args`` and look up the concrete value
        for each ``Node`` in the current execution environment.

        Args:
            args (Argument): Data structure within which to look up concrete values

            n (Node): Node to which ``args`` belongs. This is only used for error reporting.
        """

    def load_arg(n_arg: Node) -> Any:
        if n_arg not in self.env:
            raise RuntimeError(f'Node {n} referenced nonexistent value {n_arg}! Run Graph.lint() to diagnose such issues')
        return self.env[n_arg]
    return map_arg(args, load_arg)