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
def naturaldelta(value: dt.timedelta | float, months: bool=True, minimum_unit: str='seconds') -> str:
    """Return a natural representation of a timedelta or number of seconds.

    This is similar to `naturaltime`, but does not add tense to the result.

    Args:
        value (datetime.timedelta, int or float): A timedelta or a number of seconds.
        months (bool): If `True`, then a number of months (based on 30.5 days) will be
            used for fuzziness between years.
        minimum_unit (str): The lowest unit that can be used.

    Returns:
        str (str or `value`): A natural representation of the amount of time
            elapsed unless `value` is not datetime.timedelta or cannot be
            converted to int. In that case, a `value` is returned unchanged.

    Raises:
        OverflowError: If `value` is too large to convert to datetime.timedelta.

    Examples
        Compare two timestamps in a custom local timezone::

        import datetime as dt
        from dateutil.tz import gettz

        berlin = gettz("Europe/Berlin")
        now = dt.datetime.now(tz=berlin)
        later = now + dt.timedelta(minutes=30)

        assert naturaldelta(later - now) == "30 minutes"
    """
    tmp = Unit[minimum_unit.upper()]
    if tmp not in (Unit.SECONDS, Unit.MILLISECONDS, Unit.MICROSECONDS):
        msg = f"Minimum unit '{minimum_unit}' not supported"
        raise ValueError(msg)
    min_unit = tmp
    if isinstance(value, dt.timedelta):
        delta = value
    else:
        try:
            value = int(value)
            delta = dt.timedelta(seconds=value)
        except (ValueError, TypeError):
            return str(value)
    use_months = months
    seconds = abs(delta.seconds)
    days = abs(delta.days)
    years = days // 365
    days = days % 365
    num_months = int(days // 30.5)
    if not years and days < 1:
        if seconds == 0:
            if min_unit == Unit.MICROSECONDS and delta.microseconds < 1000:
                return _ngettext('%d microsecond', '%d microseconds', delta.microseconds) % delta.microseconds
            if min_unit == Unit.MILLISECONDS or (min_unit == Unit.MICROSECONDS and 1000 <= delta.microseconds < 1000000):
                milliseconds = delta.microseconds / 1000
                return _ngettext('%d millisecond', '%d milliseconds', int(milliseconds)) % milliseconds
            return _('a moment')
        if seconds == 1:
            return _('a second')
        if seconds < 60:
            return _ngettext('%d second', '%d seconds', seconds) % seconds
        if 60 <= seconds < 120:
            return _('a minute')
        if 120 <= seconds < 3600:
            minutes = seconds // 60
            return _ngettext('%d minute', '%d minutes', minutes) % minutes
        if 3600 <= seconds < 3600 * 2:
            return _('an hour')
        if 3600 < seconds:
            hours = seconds // 3600
            return _ngettext('%d hour', '%d hours', hours) % hours
    elif years == 0:
        if days == 1:
            return _('a day')
        if not use_months:
            return _ngettext('%d day', '%d days', days) % days
        if not num_months:
            return _ngettext('%d day', '%d days', days) % days
        if num_months == 1:
            return _('a month')
        return _ngettext('%d month', '%d months', num_months) % num_months
    elif years == 1:
        if not num_months and (not days):
            return _('a year')
        if not num_months:
            return _ngettext('1 year, %d day', '1 year, %d days', days) % days
        if use_months:
            if num_months == 1:
                return _('1 year, 1 month')
            return _ngettext('1 year, %d month', '1 year, %d months', num_months) % num_months
        return _ngettext('1 year, %d day', '1 year, %d days', days) % days
    return _ngettext('%d year', '%d years', years).replace('%d', '%s') % intcomma(years)