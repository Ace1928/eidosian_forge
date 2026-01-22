from typing import Any, Collection, List, Union, cast
from ...error import GraphQLError
from ...language import (
from ...type import introspection_types, specified_scalar_types
from ...pyutils import did_you_mean, suggestion_list
from . import ASTValidationRule, ValidationContext, SDLValidationContext
Known type names

    A GraphQL document is only valid if referenced types (specifically variable
    definitions and fragment conditions) are defined by the type schema.

    See https://spec.graphql.org/draft/#sec-Fragment-Spread-Type-Existence
    