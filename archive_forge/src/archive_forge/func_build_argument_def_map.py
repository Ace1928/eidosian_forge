from itertools import chain
from typing import cast, Callable, Collection, Dict, List, Union
from ..language import DirectiveLocation, parse_value
from ..pyutils import inspect, Undefined
from ..type import (
from .get_introspection_query import (
from .value_from_ast import value_from_ast
def build_argument_def_map(argument_value_introspections: Collection[IntrospectionInputValue]) -> Dict[str, GraphQLArgument]:
    return {argument_introspection['name']: build_argument(argument_introspection) for argument_introspection in argument_value_introspections}