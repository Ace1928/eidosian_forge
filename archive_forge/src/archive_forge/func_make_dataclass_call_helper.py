from collections import OrderedDict
from textwrap import dedent
import operator
from . import ExprNodes
from . import Nodes
from . import PyrexTypes
from . import Builtin
from . import Naming
from .Errors import error, warning
from .Code import UtilityCode, TempitaUtilityCode, PyxCodeWriter
from .Visitor import VisitorTransform
from .StringEncoding import EncodedString
from .TreeFragment import TreeFragment
from .ParseTreeTransforms import NormalizeTree, SkipDeclarations
from .Options import copy_inherited_directives
def make_dataclass_call_helper(pos, callable, kwds):
    utility_code = UtilityCode.load_cached('DataclassesCallHelper', 'Dataclasses.c')
    func_type = PyrexTypes.CFuncType(PyrexTypes.py_object_type, [PyrexTypes.CFuncTypeArg('callable', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('kwds', PyrexTypes.py_object_type, None)])
    return ExprNodes.PythonCapiCallNode(pos, function_name='__Pyx_DataclassesCallHelper', func_type=func_type, utility_code=utility_code, args=[callable, kwds])