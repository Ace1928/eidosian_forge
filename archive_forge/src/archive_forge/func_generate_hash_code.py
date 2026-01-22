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
def generate_hash_code(code, unsafe_hash, eq, frozen, node, fields):
    """
    Copied from CPython implementation - the intention is to follow this as far as
    is possible:
    #    +------------------- unsafe_hash= parameter
    #    |       +----------- eq= parameter
    #    |       |       +--- frozen= parameter
    #    |       |       |
    #    v       v       v    |        |        |
    #                         |   no   |  yes   |  <--- class has explicitly defined __hash__
    # +=======+=======+=======+========+========+
    # | False | False | False |        |        | No __eq__, use the base class __hash__
    # +-------+-------+-------+--------+--------+
    # | False | False | True  |        |        | No __eq__, use the base class __hash__
    # +-------+-------+-------+--------+--------+
    # | False | True  | False | None   |        | <-- the default, not hashable
    # +-------+-------+-------+--------+--------+
    # | False | True  | True  | add    |        | Frozen, so hashable, allows override
    # +-------+-------+-------+--------+--------+
    # | True  | False | False | add    | raise  | Has no __eq__, but hashable
    # +-------+-------+-------+--------+--------+
    # | True  | False | True  | add    | raise  | Has no __eq__, but hashable
    # +-------+-------+-------+--------+--------+
    # | True  | True  | False | add    | raise  | Not frozen, but hashable
    # +-------+-------+-------+--------+--------+
    # | True  | True  | True  | add    | raise  | Frozen, so hashable
    # +=======+=======+=======+========+========+
    # For boxes that are blank, __hash__ is untouched and therefore
    # inherited from the base class.  If the base is object, then
    # id-based hashing is used.

    The Python implementation creates a tuple of all the fields, then hashes them.
    This implementation creates a tuple of all the hashes of all the fields and hashes that.
    The reason for this slight difference is to avoid to-Python conversions for anything
    that Cython knows how to hash directly (It doesn't look like this currently applies to
    anything though...).
    """
    hash_entry = node.scope.lookup_here('__hash__')
    if hash_entry:
        if unsafe_hash:
            error(node.pos, 'Cannot overwrite attribute __hash__ in class %s' % node.class_name)
        return
    if not unsafe_hash:
        if not eq:
            return
        if not frozen:
            code.add_extra_statements([Nodes.SingleAssignmentNode(node.pos, lhs=ExprNodes.NameNode(node.pos, name=EncodedString('__hash__')), rhs=ExprNodes.NoneNode(node.pos))])
            return
    names = [name for name, field in fields.items() if not field.is_initvar and (field.compare.value if field.hash.value is None else field.hash.value)]
    hash_tuple_items = u', '.join((u'self.%s' % name for name in names))
    if hash_tuple_items:
        hash_tuple_items += u','
    code.add_code_lines(['def __hash__(self):', '    return hash((%s))' % hash_tuple_items])