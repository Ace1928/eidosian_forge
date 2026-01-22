from decimal import Decimal
from math import isinf, isnan
from typing import Optional, Set, SupportsFloat, Union
from xml.etree.ElementTree import Element
from elementpath import datatypes
from ..exceptions import XMLSchemaValueError
from ..translation import gettext as _
from .exceptions import XMLSchemaValidationError
def base64_binary_validator(value: Union[str, datatypes.Base64Binary]) -> None:
    if isinstance(value, datatypes.Base64Binary):
        return
    value = value.replace(' ', '')
    if not value:
        return
    match = datatypes.Base64Binary.pattern.match(value)
    if match is None or match.group(0) != value:
        raise XMLSchemaValidationError(base64_binary_validator, value, _('not a base64 encoding'))