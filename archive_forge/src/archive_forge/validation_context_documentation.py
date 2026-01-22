from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Union, cast
from ..error import GraphQLError
from ..language import (
from ..type import (
from ..utilities import TypeInfo, TypeInfoVisitor
Utility class providing a context for validation using a GraphQL schema.

    An instance of this class is passed as the context attribute to all Validators,
    allowing access to commonly useful contextual information from within a validation
    rule.
    