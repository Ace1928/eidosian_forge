from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
def members_slot_value(self, scope):
    dict_offset = self.slot_code(scope)
    if dict_offset == '0':
        return None
    return '{"__dictoffset__", T_PYSSIZET, %s, READONLY, NULL},' % dict_offset