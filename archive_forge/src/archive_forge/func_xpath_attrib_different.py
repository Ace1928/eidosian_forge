import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_attrib_different(self, xpath: XPathExpr, name: str, value: Optional[str]) -> XPathExpr:
    assert value is not None
    if value:
        xpath.add_condition('not(%s) or %s != %s' % (name, name, self.xpath_literal(value)))
    else:
        xpath.add_condition('%s != %s' % (name, self.xpath_literal(value)))
    return xpath