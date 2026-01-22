import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_function(self, function: Function) -> XPathExpr:
    """Translate a functional pseudo-class."""
    method_name = 'xpath_%s_function' % function.name.replace('-', '_')
    method = getattr(self, method_name, None)
    if not method:
        raise ExpressionError('The pseudo-class :%s() is unknown' % function.name)
    return typing.cast(XPathExpr, method(self.xpath(function.selector), function))