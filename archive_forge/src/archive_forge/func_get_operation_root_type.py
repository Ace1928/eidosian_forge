from ..error import GraphQLError
from ..language import ast
from ..pyutils.default_ordered_dict import DefaultOrderedDict
from ..type.definition import GraphQLInterfaceType, GraphQLUnionType
from ..type.directives import GraphQLIncludeDirective, GraphQLSkipDirective
from ..type.introspection import (SchemaMetaFieldDef, TypeMetaFieldDef,
from ..utils.type_from_ast import type_from_ast
from .values import get_argument_values, get_variable_values
def get_operation_root_type(schema, operation):
    op = operation.operation
    if op == 'query':
        return schema.get_query_type()
    elif op == 'mutation':
        mutation_type = schema.get_mutation_type()
        if not mutation_type:
            raise GraphQLError('Schema is not configured for mutations', [operation])
        return mutation_type
    elif op == 'subscription':
        subscription_type = schema.get_subscription_type()
        if not subscription_type:
            raise GraphQLError('Schema is not configured for subscriptions', [operation])
        return subscription_type
    raise GraphQLError('Can only execute queries, mutations and subscriptions', [operation])