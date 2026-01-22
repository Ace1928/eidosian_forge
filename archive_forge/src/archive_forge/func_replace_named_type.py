from collections import defaultdict
from typing import (
from ..language import (
from ..pyutils import inspect, merge_kwargs
from ..type import (
from .value_from_ast import value_from_ast
def replace_named_type(type_: GraphQLNamedType) -> GraphQLNamedType:
    return type_map[type_.name]