from copy import copy, deepcopy
from typing import (
from ..error import GraphQLError
from ..language import ast, OperationType
from ..pyutils import inspect, is_collection, is_description
from .definition import (
from .directives import GraphQLDirective, specified_directives, is_directive
from .introspection import introspection_types
def get_implementations(self, interface_type: GraphQLInterfaceType) -> InterfaceImplementations:
    return self._implementations_map.get(interface_type.name, InterfaceImplementations(objects=[], interfaces=[]))