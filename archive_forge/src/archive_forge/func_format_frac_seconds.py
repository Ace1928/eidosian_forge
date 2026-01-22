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
def format_frac_seconds(self, num: int) -> str:
    """ Return fractional seconds.

        Rounds the time's microseconds to the precision given by the number         of digits passed in.
        """
    value = self.value.microsecond / 1000000
    return self.format(round(value, num) * 10 ** num, num)