import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_class(self, class_selector: Class) -> XPathExpr:
    """Translate a class selector."""
    xpath = self.xpath(class_selector.selector)
    return self.xpath_attrib_includes(xpath, '@class', class_selector.class_name)