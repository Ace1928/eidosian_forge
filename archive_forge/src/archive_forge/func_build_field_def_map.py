from ..language.parser import parse_value
from ..pyutils.ordereddict import OrderedDict
from ..type import (GraphQLArgument, GraphQLBoolean, GraphQLEnumType,
from ..type.directives import DirectiveLocation, GraphQLDirective
from ..type.introspection import (TypeKind, __Directive, __DirectiveLocation,
from .value_from_ast import value_from_ast
def build_field_def_map(type_introspection):
    return OrderedDict([(f['name'], GraphQLField(type=get_output_type(f['type']), description=f.get('description'), resolver=no_execution, deprecation_reason=f.get('deprecationReason'), args=build_input_value_def_map(f.get('args'), GraphQLArgument))) for f in type_introspection.get('fields', [])])