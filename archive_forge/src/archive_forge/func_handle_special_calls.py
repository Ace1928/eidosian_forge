from pythran.analyses import Aliases
from pythran.passmanager import Transformation
from pythran.syntax import PythranSyntaxError
from pythran.tables import MODULES
import gast as ast
from copy import deepcopy
def handle_special_calls(func_alias, node):
    if func_alias is MODULES['numpy']['arange']:
        if len(node.args) == 1:
            node.args.insert(0, ast.Constant(0, None))