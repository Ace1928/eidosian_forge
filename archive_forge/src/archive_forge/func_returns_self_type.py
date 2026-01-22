from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
def returns_self_type(self):
    return self.ret_format == 'T'