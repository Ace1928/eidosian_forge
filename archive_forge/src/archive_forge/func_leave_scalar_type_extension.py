from typing import Any, Collection, Optional
from ..language.ast import Node, OperationType
from .block_string import print_block_string
from .print_string import print_string
from .visitor import visit, Visitor
@staticmethod
def leave_scalar_type_extension(node: PrintedNode, *_args: Any) -> str:
    return join(('extend scalar', node.name, join(node.directives, ' ')), ' ')