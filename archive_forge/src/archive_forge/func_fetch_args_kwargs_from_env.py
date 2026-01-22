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
def fetch_args_kwargs_from_env(self, n: Node) -> Tuple[Tuple, Dict]:
    """
        Fetch the concrete values of ``args`` and ``kwargs`` of node ``n``
        from the current execution environment.

        Args:
            n (Node): The node for which ``args`` and ``kwargs`` should be fetched.

        Return:
            Tuple[Tuple, Dict]: ``args`` and ``kwargs`` with concrete values for ``n``.
        """
    args = self.map_nodes_to_values(n.args, n)
    assert isinstance(args, tuple)
    kwargs = self.map_nodes_to_values(n.kwargs, n)
    assert isinstance(kwargs, dict)
    return (args, kwargs)