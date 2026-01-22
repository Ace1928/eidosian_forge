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
def set_up_loop(self, env):
    from . import ExprNodes
    target_type = self.target.type
    if target_type.is_numeric:
        loop_type = target_type
    else:
        if target_type.is_enum:
            warning(self.target.pos, 'Integer loops over enum values are fragile. Please cast to a safe integer type instead.')
        loop_type = PyrexTypes.c_long_type if target_type.is_pyobject else PyrexTypes.c_int_type
        if not self.bound1.type.is_pyobject:
            loop_type = PyrexTypes.widest_numeric_type(loop_type, self.bound1.type)
        if not self.bound2.type.is_pyobject:
            loop_type = PyrexTypes.widest_numeric_type(loop_type, self.bound2.type)
        if self.step is not None and (not self.step.type.is_pyobject):
            loop_type = PyrexTypes.widest_numeric_type(loop_type, self.step.type)
    self.bound1 = self.bound1.coerce_to(loop_type, env)
    self.bound2 = self.bound2.coerce_to(loop_type, env)
    if not self.bound2.is_literal:
        self.bound2 = self.bound2.coerce_to_temp(env)
    if self.step is not None:
        self.step = self.step.coerce_to(loop_type, env)
        if not self.step.is_literal:
            self.step = self.step.coerce_to_temp(env)
    if target_type.is_numeric or target_type.is_enum:
        self.is_py_target = False
        if isinstance(self.target, ExprNodes.BufferIndexNode):
            raise error(self.pos, 'Buffer or memoryview slicing/indexing not allowed as for-loop target.')
        self.loopvar_node = self.target
        self.py_loopvar_node = None
    else:
        self.is_py_target = True
        c_loopvar_node = ExprNodes.TempNode(self.pos, loop_type, env)
        self.loopvar_node = c_loopvar_node
        self.py_loopvar_node = ExprNodes.CloneNode(c_loopvar_node).coerce_to_pyobject(env)