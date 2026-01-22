import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_negation(self, negation: Negation) -> XPathExpr:
    xpath = self.xpath(negation.selector)
    sub_xpath = self.xpath(negation.subselector)
    sub_xpath.add_name_test()
    if sub_xpath.condition:
        return xpath.add_condition('not(%s)' % sub_xpath.condition)
    else:
        return xpath.add_condition('0')