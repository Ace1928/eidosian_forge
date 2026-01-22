from __future__ import annotations
import typing as T
from ...interpreterbase import (
from ...mparser import PlusAssignmentNode
def check_contains(el: T.List[TYPE_var]) -> bool:
    for element in el:
        if isinstance(element, list):
            found = check_contains(element)
            if found:
                return True
        if element == args[0]:
            return True
    return False