from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
def get_slot_function(scope, slot):
    slot_code = slot.slot_code(scope)
    if slot_code != '0':
        entry = scope.parent_scope.lookup_here(scope.parent_type.name)
        if entry.visibility != 'extern':
            return slot_code
    return None