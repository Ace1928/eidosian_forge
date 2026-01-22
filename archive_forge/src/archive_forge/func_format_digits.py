import calendar
import datetime
import decimal
import re
from typing import Any, Iterator, List, Optional, Tuple, Union
from unicodedata import category
from ..exceptions import xpath_error
from ..regex import translate_pattern
from ._translation_maps import ALPHABET_CHARACTERS, OTHER_NUMBERS, ROMAN_NUMERALS_MAP, \
def format_digits(digits: str, fmt: str, digits_family: str='0123456789', optional_digit: str='#', grouping_separator: Optional[str]=None) -> str:
    result = []
    iter_num_digits = reversed(digits)
    num_digit = next(iter_num_digits)
    for fmt_char in reversed(fmt):
        if fmt_char.isdigit() or fmt_char == optional_digit:
            if num_digit:
                result.append(digits_family[ord(num_digit) - 48])
                num_digit = next(iter_num_digits, '')
            elif fmt_char != optional_digit:
                result.append(digits_family[0])
        elif not result or (not result[-1].isdigit() and grouping_separator and (result[-1] != grouping_separator)):
            raise xpath_error('FODF1310', 'invalid grouping in picture argument')
        else:
            result.append(fmt_char)
    if num_digit:
        separator = ''
        _separator = {x for x in fmt if not x.isdigit() and x != optional_digit}
        if len(_separator) != 1:
            repeat = None
        else:
            separator = _separator.pop()
            chunks = fmt.split(separator)
            if len(chunks[0]) > len(chunks[-1]):
                repeat = None
            elif all((len(item) == len(chunks[-1]) for item in chunks[1:-1])):
                repeat = len(chunks[-1]) + 1
            else:
                repeat = None
        if repeat is None:
            while num_digit:
                result.append(digits_family[ord(num_digit) - 48])
                num_digit = next(iter_num_digits, '')
        else:
            while num_digit:
                if (len(result) + 1) % repeat == 0:
                    result.append(separator)
                result.append(digits_family[ord(num_digit) - 48])
                num_digit = next(iter_num_digits, '')
    if grouping_separator:
        return ''.join(reversed(result)).lstrip(grouping_separator)
    while result and category(result[-1]) not in ('Nd', 'Nl', 'No', 'Lu', 'Ll', 'Lt', 'Lm', 'Lo'):
        result.pop()
    return ''.join(reversed(result))