from pythran.passmanager import ModuleAnalysis
from pythran.analyses import StrictAliases, ArgumentEffects
from pythran.syntax import PythranSyntaxError
from pythran.intrinsic import ConstantIntr, FunctionIntr
from pythran import metadata
import gast as ast
def is_global_constant(node):
    if isinstance(node, ConstantIntr):
        return True
    if not isinstance(node, ast.FunctionDef):
        return False
    return metadata.get(node.body[0], metadata.StaticReturn)