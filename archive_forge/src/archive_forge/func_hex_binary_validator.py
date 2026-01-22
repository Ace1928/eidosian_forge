from decimal import Decimal
from math import isinf, isnan
from typing import Optional, Set, SupportsFloat, Union
from xml.etree.ElementTree import Element
from elementpath import datatypes
from ..exceptions import XMLSchemaValueError
from ..translation import gettext as _
from .exceptions import XMLSchemaValidationError
def hex_binary_validator(value: Union[str, datatypes.HexBinary]) -> None:
    if not isinstance(value, datatypes.HexBinary) and datatypes.HexBinary.pattern.match(value) is None:
        raise XMLSchemaValidationError(hex_binary_validator, value, _('not an hexadecimal number'))