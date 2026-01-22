from typing import Any, Callable, Dict, List, Optional, Union, cast
from ..language import print_ast, StringValueNode
from ..language.block_string import is_printable_as_block_string
from ..pyutils import inspect
from ..type import (
from .ast_from_value import ast_from_value
def print_schema_definition(schema: GraphQLSchema) -> Optional[str]:
    if schema.description is None and is_schema_of_common_names(schema):
        return None
    operation_types = []
    query_type = schema.query_type
    if query_type:
        operation_types.append(f'  query: {query_type.name}')
    mutation_type = schema.mutation_type
    if mutation_type:
        operation_types.append(f'  mutation: {mutation_type.name}')
    subscription_type = schema.subscription_type
    if subscription_type:
        operation_types.append(f'  subscription: {subscription_type.name}')
    return print_description(schema) + 'schema {\n' + '\n'.join(operation_types) + '\n}'