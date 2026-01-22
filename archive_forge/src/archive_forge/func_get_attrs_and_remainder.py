from __future__ import annotations
from typing import TYPE_CHECKING
from . import Extension
from ..treeprocessors import Treeprocessor
import re
def get_attrs_and_remainder(attrs_string: str) -> tuple[list[tuple[str, str]], str]:
    """ Parse attribute list and return a list of attribute tuples.

    Additionally, return any text that remained after a curly brace. In typical cases, its presence
    should mean that the input does not match the intended attribute list syntax.
    """
    attrs, remainder = _scanner.scan(attrs_string)
    index = remainder.find('}')
    remainder = remainder[index:] if index != -1 else ''
    return (attrs, remainder)