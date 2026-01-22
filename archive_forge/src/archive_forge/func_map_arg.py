import weakref
from typing import Dict, List
import torch
from ..decorators import mark_static_address
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, GetItemSource, GlobalWeakRefSource
from ..utils import global_key_name
from .base import MutableLocal, VariableTracker
from .constant import ConstantVariable
from .dicts import ConstDictVariable
from .lists import ListVariable
from .misc import GetAttrVariable
from .user_defined import UserDefinedObjectVariable
def map_arg(arg):
    if isinstance(arg, ConstantVariable):
        return arg.as_python_constant()
    elif isinstance(arg, ListVariable) and (not arg.items):
        return []
    elif isinstance(arg, ConstDictVariable) and isinstance(arg.source, GetItemSource) and isinstance(arg.source.base, AttrSource) and (arg.source.base.member == 'param_groups'):
        return self.value.param_groups[arg.source.index]
    raise ArgMappingException()