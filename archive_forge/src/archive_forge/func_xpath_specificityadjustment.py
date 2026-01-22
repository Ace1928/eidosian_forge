import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_specificityadjustment(self, matching: SpecificityAdjustment) -> XPathExpr:
    xpath = self.xpath(matching.selector)
    exprs = [self.xpath(selector) for selector in matching.selector_list]
    for e in exprs:
        e.add_name_test()
        if e.condition:
            xpath.add_condition(e.condition, 'or')
    return xpath