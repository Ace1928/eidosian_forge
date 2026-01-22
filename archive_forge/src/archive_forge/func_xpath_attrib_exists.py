import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_attrib_exists(self, xpath: XPathExpr, name: str, value: Optional[str]) -> XPathExpr:
    assert not value
    xpath.add_condition(name)
    return xpath