from typing import Any, Collection, Optional
from ..language.ast import Node, OperationType
from .block_string import print_block_string
from .print_string import print_string
from .visitor import visit, Visitor
@staticmethod
def leave_input_value_definition(node: PrintedNode, *_args: Any) -> str:
    return wrap('', node.description, '\n') + join((f'{node.name}: {node.type}', wrap('= ', node.default_value), join(node.directives, ' ')), ' ')