from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
def get_base_slot_function(scope, slot):
    base_type = scope.parent_type.base_type
    if base_type and scope.parent_scope is base_type.scope.parent_scope:
        parent_slot = slot.slot_code(base_type.scope)
        if parent_slot != '0':
            entry = scope.parent_scope.lookup_here(scope.parent_type.base_type.name)
            if entry.visibility != 'extern':
                return parent_slot
    return None