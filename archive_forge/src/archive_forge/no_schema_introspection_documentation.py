from typing import Any
from ....error import GraphQLError
from ....language import FieldNode
from ....type import get_named_type, is_introspection_type
from .. import ValidationRule
Prohibit introspection queries

    A GraphQL document is only valid if all fields selected are not fields that
    return an introspection type.

    Note: This rule is optional and is not part of the Validation section of the
    GraphQL Specification. This rule effectively disables introspection, which
    does not reflect best practices and should only be done if absolutely necessary.
    