from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
def is_self_arg(self, i):
    return self.fixed_arg_format[i] == 'T'