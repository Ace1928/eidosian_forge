from typing import Any, Collection, Optional
from ..language.ast import Node, OperationType
from .block_string import print_block_string
from .print_string import print_string
from .visitor import visit, Visitor
def has_multiline_items(strings: Optional[Strings]) -> bool:
    """Check whether one of the items in the list has multiple lines."""
    return any((is_multiline(item) for item in strings)) if strings else False