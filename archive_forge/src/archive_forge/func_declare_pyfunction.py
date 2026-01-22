from __future__ import absolute_import
import re
import copy
import operator
from ..Utils import try_finally_contextmanager
from .Errors import warning, error, InternalError, performance_hint
from .StringEncoding import EncodedString
from . import Options, Naming
from . import PyrexTypes
from .PyrexTypes import py_object_type, unspecified_type
from .TypeSlots import (
from . import Future
from . import Code
def declare_pyfunction(self, name, pos, allow_redefine=False):
    signature = get_property_accessor_signature(name)
    if signature:
        entry = self.declare(name, name, py_object_type, pos, 'private')
        entry.is_special = 1
        entry.signature = signature
        return entry
    else:
        error(pos, 'Only __get__, __set__ and __del__ methods allowed in a property declaration')
        return None