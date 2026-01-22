from pythran.analyses import OptimizableComprehension
from pythran.passmanager import Transformation
from pythran.transformations.normalize_tuples import ConvertToTuple
from pythran.conversion import mangle
from pythran.utils import attr_to_path, path_to_attr
import gast as ast
def make_Iterator(self, gen):
    if gen.ifs:
        ldFilter = ast.Lambda(ast.arguments([ast.Name(gen.target.id, ast.Param(), None, None)], [], None, [], [], None, []), ast.BoolOp(ast.And(), gen.ifs) if len(gen.ifs) > 1 else gen.ifs[0])
        ifilterName = ast.Attribute(value=ast.Name(id='builtins', ctx=ast.Load(), annotation=None, type_comment=None), attr='filter', ctx=ast.Load())
        return ast.Call(ifilterName, [ldFilter, gen.iter], [])
    else:
        return gen.iter