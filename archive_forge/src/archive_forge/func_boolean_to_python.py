from decimal import Decimal
from math import isinf, isnan
from typing import Optional, Set, SupportsFloat, Union
from xml.etree.ElementTree import Element
from elementpath import datatypes
from ..exceptions import XMLSchemaValueError
from ..translation import gettext as _
from .exceptions import XMLSchemaValidationError
def boolean_to_python(value: str) -> bool:
    if value in {'true', '1'}:
        return True
    elif value in {'false', '0'}:
        return False
    else:
        raise XMLSchemaValueError(_('{!r} is not a boolean value').format(value))