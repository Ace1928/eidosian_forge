import calendar
import datetime
import decimal
import re
from typing import Any, Iterator, List, Optional, Tuple, Union
from unicodedata import category
from ..exceptions import xpath_error
from ..regex import translate_pattern
from ._translation_maps import ALPHABET_CHARACTERS, OTHER_NUMBERS, ROMAN_NUMERALS_MAP, \
def decimal_to_string(value: decimal.Decimal) -> str:
    """
    Convert a Decimal value to a string representation
    that not includes exponent and with its decimals.
    """
    exponent: Any
    sign, digits, exponent = value.as_tuple()
    if not exponent:
        result = ''.join((str(x) for x in digits))
    elif exponent > 0:
        result = ''.join((str(x) for x in digits)) + '0' * exponent
    else:
        result = ''.join((str(x) for x in digits[:exponent]))
        if not result:
            result = '0'
        result += '.'
        if len(digits) >= -exponent:
            result += ''.join((str(x) for x in digits[exponent:]))
        else:
            result += '0' * (-exponent - len(digits))
            result += ''.join((str(x) for x in digits))
    return '-' + result if sign else result