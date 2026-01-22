from typing import Any, Dict, cast
from ...error import GraphQLError
from ...execution.collect_fields import collect_fields
from ...language import (
from . import ValidationRule
Subscriptions must only include a single non-introspection field.

    A GraphQL subscription is valid only if it contains a single root field and
    that root field is not an introspection field.

    See https://spec.graphql.org/draft/#sec-Single-root-field
    