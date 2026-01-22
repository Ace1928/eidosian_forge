from collections import defaultdict
from typing import (
from ..language import (
from ..pyutils import inspect, merge_kwargs
from ..type import (
from .value_from_ast import value_from_ast
def extend_field(field: GraphQLField) -> GraphQLField:
    return GraphQLField(**merge_kwargs(field.to_kwargs(), type_=replace_type(field.type), args={name: extend_arg(arg) for name, arg in field.args.items()}))