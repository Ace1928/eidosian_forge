from typing import Any, Collection, Optional
from ..language.ast import Node, OperationType
from .block_string import print_block_string
from .print_string import print_string
from .visitor import visit, Visitor
@staticmethod
def leave_interface_type_definition(node: PrintedNode, *_args: Any) -> str:
    return wrap('', node.description, '\n') + join(('interface', node.name, wrap('implements ', join(node.interfaces, ' & ')), join(node.directives, ' '), block(node.fields)), ' ')