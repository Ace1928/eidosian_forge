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
def declare_inherited_attributes(entry, base_classes):
    for base_class in base_classes:
        if base_class is PyrexTypes.error_type:
            continue
        if base_class.scope is None:
            error(pos, 'Cannot inherit from incomplete type')
        else:
            declare_inherited_attributes(entry, base_class.base_classes)
            entry.type.scope.declare_inherited_cpp_attributes(base_class)