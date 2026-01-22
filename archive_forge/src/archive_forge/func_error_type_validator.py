from decimal import Decimal
from math import isinf, isnan
from typing import Optional, Set, SupportsFloat, Union
from xml.etree.ElementTree import Element
from elementpath import datatypes
from ..exceptions import XMLSchemaValueError
from ..translation import gettext as _
from .exceptions import XMLSchemaValidationError
def error_type_validator(value: object) -> None:
    raise XMLSchemaValidationError(error_type_validator, value, _('no value is allowed for xs:error type'))