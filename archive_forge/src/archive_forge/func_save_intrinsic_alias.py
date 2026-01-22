from pythran.analyses.global_declarations import GlobalDeclarations
from pythran.intrinsic import Intrinsic, Class, UnboundValue
from pythran.passmanager import ModuleAnalysis
from pythran.tables import functions, methods, MODULES
from pythran.unparse import Unparser
from pythran.conversion import demangle
import pythran.metadata as md
from pythran.utils import isnum
from pythran.syntax import PythranSyntaxError
import gast as ast
from copy import deepcopy
from itertools import product
import io
def save_intrinsic_alias(module):
    """ Recursively save default aliases for pythonic functions. """
    for v in module.values():
        if isinstance(v, dict):
            save_intrinsic_alias(v)
        else:
            IntrinsicAliases[v] = frozenset((v,))
            if isinstance(v, Class):
                save_intrinsic_alias(v.fields)