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
@compatibility(is_backward_compatible=False)
def is_impure(self):
    """
        Returns whether this op is impure, i.e. if its op is a placeholder or
        output, or if a call_function or call_module which is impure.

        Returns:

            bool: If the op is impure or not.
        """
    if self.op in {'placeholder', 'output'}:
        return True
    if self.op == 'call_function':
        return self.target in _side_effectful_functions
    if self.op == 'call_module':
        assert self.graph.owning_module is not None, 'self.graph.owning_module not set for purity check'
        target_mod = self.graph.owning_module.get_submodule(self.target)
        assert target_mod is not None, f'Did not find expected submodule target {self.target}'
        return getattr(target_mod, '_is_impure', False)
    return False