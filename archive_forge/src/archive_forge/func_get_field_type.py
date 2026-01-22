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
def get_field_type(pos, entry):
    """
    sets the .type attribute for a field

    Returns the annotation if possible (since this is what the dataclasses
    module does). If not (for example, attributes defined with cdef) then
    it creates a string fallback.
    """
    if entry.annotation:
        return entry.annotation.string
    else:
        s = EncodedString(entry.type.declaration_code('', for_display=1))
        return ExprNodes.StringNode(pos, value=s)