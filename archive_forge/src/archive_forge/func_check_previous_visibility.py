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
def check_previous_visibility(self, entry, visibility, pos):
    if entry.visibility != visibility:
        error(pos, "'%s' previously declared as '%s'" % (entry.name, entry.visibility))