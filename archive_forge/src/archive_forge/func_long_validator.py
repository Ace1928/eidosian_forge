from decimal import Decimal
from math import isinf, isnan
from typing import Optional, Set, SupportsFloat, Union
from xml.etree.ElementTree import Element
from elementpath import datatypes
from ..exceptions import XMLSchemaValueError
from ..translation import gettext as _
from .exceptions import XMLSchemaValidationError
def long_validator(value: int) -> None:
    if not -2 ** 63 <= value < 2 ** 63:
        raise XMLSchemaValidationError(long_validator, value, _('value must be {:s}').format('-2^63 <= x < 2^63'))