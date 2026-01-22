from typing import Any, Dict
from ...error import GraphQLError
from ...language import NameNode, OperationDefinitionNode, VisitorAction, SKIP
from . import ASTValidationContext, ASTValidationRule
Unique operation names

    A GraphQL document is only valid if all defined operations have unique names.

    See https://spec.graphql.org/draft/#sec-Operation-Name-Uniqueness
    