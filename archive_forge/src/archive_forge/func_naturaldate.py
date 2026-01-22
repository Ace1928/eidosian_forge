from __future__ import annotations
import collections.abc
import datetime as dt
import math
import typing
from enum import Enum
from functools import total_ordering
from typing import Any
from .i18n import _gettext as _
from .i18n import _ngettext
from .number import intcomma
def naturaldate(value: dt.date | dt.datetime) -> str:
    """Like `naturalday`, but append a year for dates more than ~five months away."""
    try:
        value = dt.date(value.year, value.month, value.day)
    except AttributeError:
        return str(value)
    except (OverflowError, ValueError):
        return str(value)
    delta = _abs_timedelta(value - dt.date.today())
    if delta.days >= 5 * 365 / 12:
        return naturalday(value, '%b %d %Y')
    return naturalday(value)