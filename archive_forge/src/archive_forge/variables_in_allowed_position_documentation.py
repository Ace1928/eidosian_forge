from typing import Any, Dict, Optional, cast
from ...error import GraphQLError
from ...language import (
from ...pyutils import Undefined
from ...type import GraphQLNonNull, GraphQLSchema, GraphQLType, is_non_null_type
from ...utilities import type_from_ast, is_type_sub_type_of
from . import ValidationContext, ValidationRule
Check for allowed variable usage.

    Returns True if the variable is allowed in the location it was found, which includes
    considering if default values exist for either the variable or the location at which
    it is located.
    