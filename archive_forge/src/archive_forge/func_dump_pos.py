from __future__ import absolute_import
import cython
import sys, copy
from itertools import chain
from . import Builtin
from .Errors import error, warning, InternalError, CompileError, CannotSpecialize
from . import Naming
from . import PyrexTypes
from . import TypeSlots
from .PyrexTypes import py_object_type, error_type
from .Symtab import (ModuleScope, LocalScope, ClosureScope, PropertyScope,
from .Code import UtilityCode
from .StringEncoding import EncodedString
from . import Future
from . import Options
from . import DebugFlags
from .Pythran import has_np_pythran, pythran_type, is_pythran_buffer
from ..Utils import add_metaclass, str_to_number
def dump_pos(self, mark_column=False, marker='(#)'):
    """Debug helper method that returns the source code context of this node as a string.
        """
    if not self.pos:
        return u''
    source_desc, line, col = self.pos
    contents = source_desc.get_lines(encoding='ASCII', error_handling='ignore')
    lines = contents[max(0, line - 3):line]
    current = lines[-1]
    if mark_column:
        current = current[:col] + marker + current[col:]
    lines[-1] = current.rstrip() + u'             # <<<<<<<<<<<<<<\n'
    lines += contents[line:line + 2]
    return u'"%s":%d:%d\n%s\n' % (source_desc.get_escaped_description(), line, col, u''.join(lines))