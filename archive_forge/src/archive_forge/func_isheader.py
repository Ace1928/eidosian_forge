from __future__ import annotations
from typing import TYPE_CHECKING
from . import Extension
from ..treeprocessors import Treeprocessor
import re
def isheader(elem: Element) -> bool:
    return elem.tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']