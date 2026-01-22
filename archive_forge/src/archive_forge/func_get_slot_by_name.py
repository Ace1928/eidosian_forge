from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
def get_slot_by_name(slot_name, compiler_directives):
    for slot in get_slot_table(compiler_directives).slot_table:
        if slot.slot_name == slot_name:
            return slot
    assert False, 'Slot not found: %s' % slot_name