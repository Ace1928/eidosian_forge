from __future__ import annotations
import decimal
import gettext
import locale
import os
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Callable, Iterable
from babel.core import Locale
from babel.dates import format_date, format_datetime, format_time, format_timedelta
from babel.numbers import (
def udnpgettext(self, domain: str, context: str, singular: str, plural: str, num: int) -> str:
    """Like ``unpgettext``, but look the message up in the specified
        `domain`.
        """
    return self._domains.get(domain, self).unpgettext(context, singular, plural, num)