import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_relation_descendant_combinator(self, left: XPathExpr, right: XPathExpr) -> XPathExpr:
    """right is a child, grand-child or further descendant of left; select left"""
    return left.join('[descendant::', right, closing_combiner=']', has_inner_condition=True)