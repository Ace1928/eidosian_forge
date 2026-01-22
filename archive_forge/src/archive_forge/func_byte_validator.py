from decimal import Decimal
from math import isinf, isnan
from typing import Optional, Set, SupportsFloat, Union
from xml.etree.ElementTree import Element
from elementpath import datatypes
from ..exceptions import XMLSchemaValueError
from ..translation import gettext as _
from .exceptions import XMLSchemaValidationError
def byte_validator(value: int) -> None:
    if not -2 ** 7 <= value < 2 ** 7:
        raise XMLSchemaValidationError(int_validator, value, _('value must be {:s}').format('-128 <= x < 128'))