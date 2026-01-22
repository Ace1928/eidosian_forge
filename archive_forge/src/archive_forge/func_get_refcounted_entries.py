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
def get_refcounted_entries(self, include_weakref=False, include_gc_simple=True):
    py_attrs = []
    py_buffers = []
    memoryview_slices = []
    for entry in self.var_entries:
        if entry.type.is_pyobject:
            if include_weakref or (self.is_closure_class_scope or entry.name != '__weakref__'):
                if include_gc_simple or not entry.type.is_gc_simple:
                    py_attrs.append(entry)
        elif entry.type == PyrexTypes.c_py_buffer_type:
            py_buffers.append(entry)
        elif entry.type.is_memoryviewslice:
            memoryview_slices.append(entry)
    have_entries = py_attrs or py_buffers or memoryview_slices
    return (have_entries, (py_attrs, py_buffers, memoryview_slices))