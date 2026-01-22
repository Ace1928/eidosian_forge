import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_root_pseudo(self, xpath: XPathExpr) -> XPathExpr:
    return xpath.add_condition('not(parent::*)')