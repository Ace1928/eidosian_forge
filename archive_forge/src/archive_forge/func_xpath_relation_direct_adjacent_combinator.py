import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_relation_direct_adjacent_combinator(self, left: XPathExpr, right: XPathExpr) -> XPathExpr:
    """right is a sibling immediately after left; select left"""
    xpath = left.add_condition("following-sibling::*[(name() = '{}') and (position() = 1)]".format(right.element))
    return xpath