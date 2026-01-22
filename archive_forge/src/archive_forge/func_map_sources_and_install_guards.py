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
def map_sources_and_install_guards(self, tx):
    from .builder import VariableBuilder
    self.grad_to_source = {}
    self.tensor_to_source = {}
    for g_ind, group in enumerate(self.value.param_groups):
        group_source = GetItemSource(AttrSource(self.source, 'param_groups'), g_ind)
        for p_ind, p in enumerate(group['params']):
            param_source = GetItemSource(GetItemSource(group_source, 'params'), p_ind)
            self.tensor_to_source[p] = param_source
            if p.grad is not None:
                self.grad_to_source[p.grad] = AttrSource(param_source, 'grad')
    state_source = AttrSource(self.source, 'state')
    install_guard(state_source.make_guard(GuardBuilder.DICT_KEYS))
    for p, value in self.value.state.items():
        tx.store_global_weakref(global_key_name(p), p)
        p_state_source = GetItemSource(state_source, self.tensor_to_source[p])
        install_guard(p_state_source.make_guard(GuardBuilder.DICT_KEYS))
        for k, v in value.items():
            if isinstance(v, torch.Tensor) and v not in self.grad_to_source and (v not in self.tensor_to_source):
                self.tensor_to_source[v] = GetItemSource(p_state_source, k)
            elif v is None or isinstance(v, (bool, int, float, str)):
                install_guard(GetItemSource(p_state_source, k).make_guard(GuardBuilder.CONSTANT_MATCH))
            else:
                raise GuardInstallException()
    VariableBuilder(tx, AttrSource(self.source, 'param_groups'))(self.value.param_groups).recursive_realize()