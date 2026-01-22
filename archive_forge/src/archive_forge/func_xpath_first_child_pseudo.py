import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_first_child_pseudo(self, xpath: XPathExpr) -> XPathExpr:
    return xpath.add_condition('count(preceding-sibling::*) = 0')