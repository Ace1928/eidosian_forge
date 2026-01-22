import calendar
import datetime
import decimal
import re
from typing import Any, Iterator, List, Optional, Tuple, Union
from unicodedata import category
from ..exceptions import xpath_error
from ..regex import translate_pattern
from ._translation_maps import ALPHABET_CHARACTERS, OTHER_NUMBERS, ROMAN_NUMERALS_MAP, \
def to_ordinal_it(num_as_words: str, fmt_modifier: str) -> str:
    if '%spellout-ordinal-feminine' in fmt_modifier:
        suffix = 'a'
    elif fmt_modifier.startswith('o(-'):
        suffix = fmt_modifier[3:-1]
    else:
        suffix = ''
    ordinal_map = {'zero': '', 'uno': 'primo', 'due': 'secondo', 'tre': 'terzo', 'quattro': 'quarto', 'cinque': 'quinto', 'sei': 'sesto', 'sette': 'settimo', 'otto': 'ottavo', 'nove': 'nono', 'dieci': 'decimo'}
    try:
        value = ordinal_map[num_as_words]
    except KeyError:
        if num_as_words[-1] in 'eo':
            value = num_as_words[:-1] + 'esimo'
        else:
            value = num_as_words + 'esimo'
    if value and suffix:
        return value[:-1] + suffix
    return value