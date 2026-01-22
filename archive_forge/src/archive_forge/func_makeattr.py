from pythran.analyses import OptimizableComprehension
from pythran.passmanager import Transformation
from pythran.transformations.normalize_tuples import ConvertToTuple
from pythran.conversion import mangle
from pythran.utils import attr_to_path, path_to_attr
import gast as ast
def makeattr(*args):
    return ast.Call(ast.Attribute(value=ast.Name(id='builtins', ctx=ast.Load(), annotation=None, type_comment=None), attr='map', ctx=ast.Load()), list(args), [])