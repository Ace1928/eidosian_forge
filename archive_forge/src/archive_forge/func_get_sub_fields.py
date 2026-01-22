from ..error import GraphQLError
from ..language import ast
from ..pyutils.default_ordered_dict import DefaultOrderedDict
from ..type.definition import GraphQLInterfaceType, GraphQLUnionType
from ..type.directives import GraphQLIncludeDirective, GraphQLSkipDirective
from ..type.introspection import (SchemaMetaFieldDef, TypeMetaFieldDef,
from ..utils.type_from_ast import type_from_ast
from .values import get_argument_values, get_variable_values
def get_sub_fields(self, return_type, field_asts):
    k = (return_type, tuple(field_asts))
    if k not in self._subfields_cache:
        subfield_asts = DefaultOrderedDict(list)
        visited_fragment_names = set()
        for field_ast in field_asts:
            selection_set = field_ast.selection_set
            if selection_set:
                subfield_asts = collect_fields(self, return_type, selection_set, subfield_asts, visited_fragment_names)
        self._subfields_cache[k] = subfield_asts
    return self._subfields_cache[k]