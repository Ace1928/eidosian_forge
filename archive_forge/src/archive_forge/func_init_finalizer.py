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
def init_finalizer(gm):

    def clear_static_tensor_refs():
        for name in names_to_delete:
            gm._buffers.pop(name, None)
            gm._parameters.pop(name, None)
            if tc.params_flat:
                tc.params_flat.clear()
    weakref.finalize(value, clear_static_tensor_refs)