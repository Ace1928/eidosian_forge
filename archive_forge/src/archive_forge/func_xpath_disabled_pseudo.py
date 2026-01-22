import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_disabled_pseudo(self, xpath: XPathExpr) -> XPathExpr:
    return xpath.add_condition("\n        (\n            @disabled and\n            (\n                (name(.) = 'input' and @type != 'hidden') or\n                name(.) = 'button' or\n                name(.) = 'select' or\n                name(.) = 'textarea' or\n                name(.) = 'command' or\n                name(.) = 'fieldset' or\n                name(.) = 'optgroup' or\n                name(.) = 'option'\n            )\n        ) or (\n            (\n                (name(.) = 'input' and @type != 'hidden') or\n                name(.) = 'button' or\n                name(.) = 'select' or\n                name(.) = 'textarea'\n            )\n            and ancestor::fieldset[@disabled]\n        )\n        ")