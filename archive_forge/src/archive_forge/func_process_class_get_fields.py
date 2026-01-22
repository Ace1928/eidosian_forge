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
def process_class_get_fields(node):
    var_entries = node.scope.var_entries
    var_entries = sorted(var_entries, key=operator.attrgetter('pos'))
    var_names = [entry.name for entry in var_entries]
    transform = RemoveAssignmentsToNames(var_names)
    transform(node)
    default_value_assignments = transform.removed_assignments
    base_type = node.base_type
    fields = OrderedDict()
    while base_type:
        if base_type.is_external or not base_type.scope.implemented:
            warning(node.pos, 'Cannot reliably handle Cython dataclasses with base types in external modules since it is not possible to tell what fields they have', 2)
        if base_type.dataclass_fields:
            fields = base_type.dataclass_fields.copy()
            break
        base_type = base_type.base_type
    for entry in var_entries:
        name = entry.name
        is_initvar = entry.declared_with_pytyping_modifier('dataclasses.InitVar')
        is_classvar = entry.declared_with_pytyping_modifier('typing.ClassVar')
        if name in default_value_assignments:
            assignment = default_value_assignments[name]
            if isinstance(assignment, ExprNodes.CallNode) and (assignment.function.as_cython_attribute() == 'dataclasses.field' or Builtin.exprnode_to_known_standard_library_name(assignment.function, node.scope) == 'dataclasses.field'):
                valid_general_call = isinstance(assignment, ExprNodes.GeneralCallNode) and isinstance(assignment.positional_args, ExprNodes.TupleNode) and (not assignment.positional_args.args) and (assignment.keyword_args is None or isinstance(assignment.keyword_args, ExprNodes.DictNode))
                valid_simple_call = isinstance(assignment, ExprNodes.SimpleCallNode) and (not assignment.args)
                if not (valid_general_call or valid_simple_call):
                    error(assignment.pos, "Call to 'cython.dataclasses.field' must only consist of compile-time keyword arguments")
                    continue
                keyword_args = assignment.keyword_args.as_python_dict() if valid_general_call and assignment.keyword_args else {}
                if 'default' in keyword_args and 'default_factory' in keyword_args:
                    error(assignment.pos, 'cannot specify both default and default_factory')
                    continue
                field = Field(node.pos, **keyword_args)
            else:
                if assignment.type in [Builtin.list_type, Builtin.dict_type, Builtin.set_type]:
                    error(assignment.pos, "mutable default <class '{0}'> for field {1} is not allowed: use default_factory".format(assignment.type.name, name))
                field = Field(node.pos, default=assignment)
        else:
            field = Field(node.pos)
        field.is_initvar = is_initvar
        field.is_classvar = is_classvar
        if entry.visibility == 'private':
            field.private = True
        fields[name] = field
    node.entry.type.dataclass_fields = fields
    return fields