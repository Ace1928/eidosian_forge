from operator import attrgetter
from typing import Any, Collection
from ...error import GraphQLError
from ...language import ArgumentNode, DirectiveNode, FieldNode
from ...pyutils import group_by
from . import ASTValidationRule
Unique argument names

    A GraphQL field or directive is only valid if all supplied arguments are uniquely
    named.

    See https://spec.graphql.org/draft/#sec-Argument-Names
    