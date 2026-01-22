from typing import Any, Collection, Optional
from ..language.ast import Node, OperationType
from .block_string import print_block_string
from .print_string import print_string
from .visitor import visit, Visitor
@staticmethod
def leave_field_definition(node: PrintedNode, *_args: Any) -> str:
    args = node.arguments
    args = wrap('(\n', indent(join(args, '\n')), '\n)') if has_multiline_items(args) else wrap('(', join(args, ', '), ')')
    directives = wrap(' ', join(node.directives, ' '))
    return wrap('', node.description, '\n') + f'{node.name}{args}: {node.type}{directives}'