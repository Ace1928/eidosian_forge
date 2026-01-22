from typing import Any, Callable, Dict, List, Optional, Union, cast
from ..language import print_ast, StringValueNode
from ..language.block_string import is_printable_as_block_string
from ..pyutils import inspect
from ..type import (
from .ast_from_value import ast_from_value
def is_schema_of_common_names(schema: GraphQLSchema) -> bool:
    """Check whether this schema uses the common naming convention.

    GraphQL schema define root types for each type of operation. These types are the
    same as any other type and can be named in any manner, however there is a common
    naming convention:

    schema {
      query: Query
      mutation: Mutation
      subscription: Subscription
    }

    When using this naming convention, the schema description can be omitted.
    """
    query_type = schema.query_type
    if query_type and query_type.name != 'Query':
        return False
    mutation_type = schema.mutation_type
    if mutation_type and mutation_type.name != 'Mutation':
        return False
    subscription_type = schema.subscription_type
    return not subscription_type or subscription_type.name == 'Subscription'