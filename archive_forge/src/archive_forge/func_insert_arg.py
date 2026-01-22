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
def insert_arg(self, idx: int, arg: Argument) -> None:
    """
        Insert an positional argument to the argument list with given index.

        Args:

            idx (int): The index of the element in ``self.args`` to be inserted before.
            arg (Argument): The new argument value to insert into ``args``
        """
    assert 0 <= idx <= len(self.args), 'insert_args index must be between 0 and len(self.args)'
    args_left = self.args[:idx]
    args_right = self.args[idx:]
    self._args = args_left + (arg,) + args_right
    _new_input_nodes = {}
    map_arg(arg, _new_input_nodes.setdefault)
    for new_use in _new_input_nodes.keys():
        if new_use not in self._input_nodes:
            self._input_nodes.setdefault(new_use)
            new_use.users.setdefault(self)