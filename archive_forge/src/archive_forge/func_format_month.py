from __future__ import annotations
import re
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, SupportsInt
import datetime
from collections.abc import Iterable
from babel import localtime
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def format_month(self, char: str, num: int) -> str:
    if num <= 2:
        return '%0*d' % (num, self.value.month)
    width = {3: 'abbreviated', 4: 'wide', 5: 'narrow'}[num]
    context = {'M': 'format', 'L': 'stand-alone'}[char]
    return get_month_names(width, context, self.locale)[self.value.month]