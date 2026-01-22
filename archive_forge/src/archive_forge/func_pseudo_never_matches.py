import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def pseudo_never_matches(self, xpath: XPathExpr) -> XPathExpr:
    """Common implementation for pseudo-classes that never match."""
    return xpath.add_condition('0')