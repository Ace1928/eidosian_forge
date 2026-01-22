from decimal import Decimal
from math import isinf, isnan
from typing import Optional, Set, SupportsFloat, Union
from xml.etree.ElementTree import Element
from elementpath import datatypes
from ..exceptions import XMLSchemaValueError
from ..translation import gettext as _
from .exceptions import XMLSchemaValidationError
def unsigned_int_validator(value: int) -> None:
    if not 0 <= value < 2 ** 32:
        raise XMLSchemaValidationError(unsigned_int_validator, value, _('value must be {:s}').format('0 <= x < 2^32'))