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
def format_era(self, char: str, num: int) -> str:
    width = {3: 'abbreviated', 4: 'wide', 5: 'narrow'}[max(3, num)]
    era = int(self.value.year >= 0)
    return get_era_names(width, self.locale)[era]