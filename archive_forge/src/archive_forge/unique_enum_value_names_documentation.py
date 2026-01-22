from collections import defaultdict
from typing import cast, Any, Dict
from ...error import GraphQLError
from ...language import NameNode, EnumTypeDefinitionNode, VisitorAction, SKIP
from ...type import is_enum_type, GraphQLEnumType
from . import SDLValidationContext, SDLValidationRule
Unique enum value names

    A GraphQL enum type is only valid if all its values are uniquely named.
    