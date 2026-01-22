from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
def format_from_type(self, arg_type):
    if arg_type.is_pyobject:
        arg_type = PyrexTypes.py_object_type
    return self.type_to_format_map[arg_type]