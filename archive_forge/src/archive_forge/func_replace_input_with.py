from typing import TYPE_CHECKING, Union, Callable, Any, Tuple, List, Optional, Dict, Set
from ._compatibility import compatibility
from .immutable_collections import immutable_dict, immutable_list
import torch
import builtins
import types
import inspect
import warnings
from torch.fx.operator_schemas import normalize_function, normalize_module, ArgsKwargsPair
from .._ops import ops as _ops
@compatibility(is_backward_compatible=True)
def replace_input_with(self, old_input: 'Node', new_input: 'Node'):
    """
        Loop through input nodes of ``self``, and replace all instances of
        ``old_input`` with ``new_input``.

        Args:

            old_input (Node): The old input node to be replaced.
            new_input (Node): The new input node to replace ``old_input``.
        """

    def maybe_replace_node(n: Node) -> Node:
        return new_input if n == old_input else n
    new_args = map_arg(self.args, maybe_replace_node)
    new_kwargs = map_arg(self.kwargs, maybe_replace_node)
    assert isinstance(new_args, tuple)
    assert isinstance(new_kwargs, dict)
    self.__update_args_kwargs(new_args, new_kwargs)