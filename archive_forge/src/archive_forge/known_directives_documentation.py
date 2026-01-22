from typing import cast, Any, Dict, List, Optional, Tuple, Union
from ...error import GraphQLError
from ...language import (
from ...type import specified_directives
from . import ASTValidationRule, SDLValidationContext, ValidationContext
Known directives

    A GraphQL document is only valid if all ``@directives`` are known by the schema and
    legally positioned.

    See https://spec.graphql.org/draft/#sec-Directives-Are-Defined
    