from decimal import Decimal
from math import isinf, isnan
from typing import Optional, Set, SupportsFloat, Union
from xml.etree.ElementTree import Element
from elementpath import datatypes
from ..exceptions import XMLSchemaValueError
from ..translation import gettext as _
from .exceptions import XMLSchemaValidationError
def negative_int_validator(value: int) -> None:
    if value >= 0:
        raise XMLSchemaValidationError(negative_int_validator, value, _('value must be negative'))