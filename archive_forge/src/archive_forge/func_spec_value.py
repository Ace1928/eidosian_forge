from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
def spec_value(self, scope):
    if self.is_empty(scope):
        return '0'
    return self.substructure_cname(scope)