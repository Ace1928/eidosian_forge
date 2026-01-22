from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def get_decimal_precision(number: decimal.Decimal) -> int:
    """Return maximum precision of a decimal instance's fractional part.

    Precision is extracted from the fractional part only.
    """
    assert isinstance(number, decimal.Decimal)
    decimal_tuple = number.normalize().as_tuple()
    if not isinstance(decimal_tuple.exponent, int) or decimal_tuple.exponent >= 0:
        return 0
    return abs(decimal_tuple.exponent)