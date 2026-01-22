import calendar
import datetime
import decimal
import re
from typing import Any, Iterator, List, Optional, Tuple, Union
from unicodedata import category
from ..exceptions import xpath_error
from ..regex import translate_pattern
from ._translation_maps import ALPHABET_CHARACTERS, OTHER_NUMBERS, ROMAN_NUMERALS_MAP, \
def int_to_month(num: int, lang: Optional[str]=None) -> str:
    if lang is None:
        lang = 'en'
    try:
        months_map = NUM_TO_MONTH_MAPS[lang]
    except KeyError:
        months_map = NUM_TO_MONTH_MAPS['en']
    return months_map[num]