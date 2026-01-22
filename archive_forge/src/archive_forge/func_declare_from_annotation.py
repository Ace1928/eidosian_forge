from __future__ import absolute_import
import cython
import re
import sys
import copy
import os.path
import operator
from .Errors import (
from .Code import UtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from . import Nodes
from .Nodes import Node, utility_code_for_imports, SingleAssignmentNode
from . import PyrexTypes
from .PyrexTypes import py_object_type, typecast, error_type, \
from . import TypeSlots
from .Builtin import (
from . import Builtin
from . import Symtab
from .. import Utils
from .Annotate import AnnotationItem
from . import Future
from ..Debugging import print_call_chain
from .DebugFlags import debug_disposal_code, debug_coercion
from .Pythran import (to_pythran, is_pythran_supported_type, is_pythran_supported_operation_type,
from .PyrexTypes import PythranExpr
def declare_from_annotation(self, env, as_target=False):
    """Implements PEP 526 annotation typing in a fairly relaxed way.

        Annotations are ignored for global variables.
        All other annotations are stored on the entry in the symbol table.
        String literals are allowed and not evaluated.
        The ambiguous Python types 'int' and 'long' are not evaluated - the 'cython.int' form must be used instead.
        """
    name = self.name
    annotation = self.annotation
    entry = self.entry or env.lookup_here(name)
    if not entry:
        if env.is_module_scope:
            return
        modifiers = ()
        if annotation.expr.is_string_literal or not env.directives['annotation_typing']:
            atype = None
        elif env.is_py_class_scope:
            atype = py_object_type
        else:
            modifiers, atype = annotation.analyse_type_annotation(env)
        if atype is None:
            atype = unspecified_type if as_target and env.directives['infer_types'] != False else py_object_type
        elif atype.is_fused and env.fused_to_specific:
            try:
                atype = atype.specialize(env.fused_to_specific)
            except CannotSpecialize:
                error(self.pos, "'%s' cannot be specialized since its type is not a fused argument to this function" % self.name)
                atype = error_type
        visibility = 'private'
        if env.is_c_dataclass_scope:
            is_frozen = env.is_c_dataclass_scope == 'frozen'
            if atype.is_pyobject or atype.can_coerce_to_pyobject(env):
                visibility = 'readonly' if is_frozen else 'public'
        if as_target and env.is_c_class_scope and (not (atype.is_pyobject or atype.is_error)):
            atype = py_object_type
            warning(annotation.pos, 'Annotation ignored since class-level attributes must be Python objects. Were you trying to set up an instance attribute?', 2)
        entry = self.entry = env.declare_var(name, atype, self.pos, is_cdef=not as_target, visibility=visibility, pytyping_modifiers=modifiers)
    if annotation and (not entry.annotation):
        entry.annotation = annotation