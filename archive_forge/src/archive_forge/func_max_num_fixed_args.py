from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
def max_num_fixed_args(self):
    return len(self.fixed_arg_format)