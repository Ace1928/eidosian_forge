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
def format_year(self, char: str, num: int) -> str:
    value = self.value.year
    if char.isupper():
        value = self.value.isocalendar()[0]
    year = self.format(value, num)
    if num == 2:
        year = year[-2:]
    return year