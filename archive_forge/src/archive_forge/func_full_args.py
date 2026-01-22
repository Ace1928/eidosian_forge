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
def full_args(func, call):
    args = call.args
    if isinstance(func, ast.FunctionDef):
        extra = len(func.args.args) - len(args)
        if extra:
            tail = [deepcopy(n) for n in func.args.defaults[extra:]]
            for arg in tail:
                self.visit(arg)
            args = args + tail
    return args