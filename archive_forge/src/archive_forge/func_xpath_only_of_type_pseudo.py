import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_only_of_type_pseudo(self, xpath: XPathExpr) -> XPathExpr:
    if xpath.element == '*':
        raise ExpressionError('*:only-of-type is not implemented')
    return xpath.add_condition('count(parent::*/child::%s) = 1' % xpath.element)