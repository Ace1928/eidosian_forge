from typing import Any, List
from ...error import GraphQLError
from ...language import (
from . import ASTValidationContext, ASTValidationRule
No unused fragments

    A GraphQL document is only valid if all fragment definitions are spread within
    operations, or spread within other fragments spread within operations.

    See https://spec.graphql.org/draft/#sec-Fragments-Must-Be-Used
    