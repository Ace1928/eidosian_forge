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
def parametrize(exp):
    if isinstance(exp, (ast.Constant, Intrinsic, ast.FunctionDef)):
        return lambda _: {exp}
    elif isinstance(exp, ContainerOf):
        index = exp.index
        return lambda args: {ContainerOf({pc for containee in exp.containees for pc in parametrize(containee)(args)}, index)}
    elif isinstance(exp, ast.Name):
        try:
            w = node.args.args.index(exp)

            def return_alias(args):
                if w < len(args):
                    return {args[w]}
                else:
                    return {node.args.defaults[w - len(args)]}
            return return_alias
        except ValueError:
            return lambda _: self.get_unbound_value_set()
    elif isinstance(exp, ast.Subscript):
        values = parametrize(exp.value)
        slices = parametrize(exp.slice)
        return lambda args: {ast.Subscript(value, slice, ast.Load()) for value in values(args) for slice in slices(args)}
    else:
        return lambda _: self.get_unbound_value_set()