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
def declare_property(self, name, doc, pos, ctype=None, property_scope=None):
    entry = self.lookup_here(name)
    if entry is None:
        entry = self.declare(name, name, py_object_type if ctype is None else ctype, pos, 'private')
    entry.is_property = True
    if ctype is not None:
        entry.is_cproperty = True
    entry.doc = doc
    if property_scope is None:
        entry.scope = PropertyScope(name, class_scope=self)
    else:
        entry.scope = property_scope
    self.property_entries.append(entry)
    return entry