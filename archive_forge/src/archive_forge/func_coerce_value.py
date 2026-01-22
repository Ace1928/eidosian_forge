from collections.abc import Iterable
import json
from ..error import GraphQLError
from ..language.printer import print_ast
from ..type import (GraphQLEnumType, GraphQLInputObjectType, GraphQLList,
from ..utils.is_valid_value import is_valid_value
from ..utils.type_from_ast import type_from_ast
from ..utils.value_from_ast import value_from_ast
def coerce_value(type, value):
    """Given a type and any value, return a runtime value coerced to match the type."""
    if isinstance(type, GraphQLNonNull):
        return coerce_value(type.of_type, value)
    if value is None:
        return None
    if isinstance(type, GraphQLList):
        item_type = type.of_type
        if not isinstance(value, str) and isinstance(value, Iterable):
            return [coerce_value(item_type, item) for item in value]
        else:
            return [coerce_value(item_type, value)]
    if isinstance(type, GraphQLInputObjectType):
        fields = type.fields
        obj = {}
        for field_name, field in fields.items():
            field_value = coerce_value(field.type, value.get(field_name))
            if field_value is None:
                field_value = field.default_value
            if field_value is not None:
                obj[field.out_name or field_name] = field_value
        return obj
    assert isinstance(type, (GraphQLScalarType, GraphQLEnumType)), 'Must be input type'
    return type.parse_value(value)