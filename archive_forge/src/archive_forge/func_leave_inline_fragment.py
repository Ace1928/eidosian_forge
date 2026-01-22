from typing import Any, Collection, Optional
from ..language.ast import Node, OperationType
from .block_string import print_block_string
from .print_string import print_string
from .visitor import visit, Visitor
@staticmethod
def leave_inline_fragment(node: PrintedNode, *_args: Any) -> str:
    return join(('...', wrap('on ', node.type_condition), join(node.directives, ' '), node.selection_set), ' ')