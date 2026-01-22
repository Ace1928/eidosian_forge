from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
def get_member_specs(self, scope):
    return [get_slot_by_name('tp_dictoffset', scope.directives).members_slot_value(scope)]