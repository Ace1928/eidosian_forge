import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_first_of_type_pseudo(self, xpath: XPathExpr) -> XPathExpr:
    if xpath.element == '*':
        raise ExpressionError('*:first-of-type is not implemented')
    return xpath.add_condition('count(preceding-sibling::%s) = 0' % xpath.element)