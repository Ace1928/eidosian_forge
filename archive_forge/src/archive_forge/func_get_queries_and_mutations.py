from typing import Callable, Dict, List, Optional, Union, Tuple
from graphql import GraphQLError
from graphql.validation import ValidationContext, ValidationRule
from graphql.language import (
from ..utils.is_introspection_key import is_introspection_key
def get_queries_and_mutations(definitions: Tuple[DefinitionNode, ...]) -> Dict[str, OperationDefinitionNode]:
    operations = {}
    for definition in definitions:
        if isinstance(definition, OperationDefinitionNode):
            operation = definition.name.value if definition.name else 'anonymous'
            operations[operation] = definition
    return operations