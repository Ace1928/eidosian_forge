import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_attrib_dashmatch(self, xpath: XPathExpr, name: str, value: Optional[str]) -> XPathExpr:
    assert value is not None
    xpath.add_condition('%s and (%s = %s or starts-with(%s, %s))' % (name, name, self.xpath_literal(value), name, self.xpath_literal(value + '-')))
    return xpath