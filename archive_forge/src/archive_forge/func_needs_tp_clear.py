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
def needs_tp_clear(self):
    """
        Do we need to generate an implementation for the tp_clear slot? Can
        be disabled to keep references for the __dealloc__ cleanup function.
        """
    return self.needs_gc() and (not self.directives.get('no_gc_clear', False))