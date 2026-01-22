import inspect
from typing import Dict, List
import torch.utils._pytree as pytree
from torch.overrides import _get_overloaded_args, get_default_nowrap_functions
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, GlobalSource
from ..utils import is_tensor_base_attr_getter
from .base import VariableTracker
from .constant import ConstantVariable
from .lists import TupleVariable
from .tensor import TensorVariable
from .user_defined import UserDefinedClassVariable
def build_torch_function_fn(tx, value, source):
    from .builder import SourcelessBuilder, VariableBuilder
    if not source:
        return VariableBuilder(tx, AttrSource(AttrSource(source, '__torch_function__'), '__func__'))(value.__torch_function__.__func__)
    else:
        return SourcelessBuilder()(tx, value.__torch_function__.__func__)