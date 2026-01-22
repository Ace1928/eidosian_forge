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
def precisedelta(value: dt.timedelta | int | None, minimum_unit: str='seconds', suppress: typing.Iterable[str]=(), format: str='%0.2f') -> str:
    """Return a precise representation of a timedelta.

    ```pycon
    >>> import datetime as dt
    >>> from humanize.time import precisedelta

    >>> delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
    >>> precisedelta(delta)
    '2 days, 1 hour and 33.12 seconds'

    ```

    A custom `format` can be specified to control how the fractional part
    is represented:

    ```pycon
    >>> precisedelta(delta, format="%0.4f")
    '2 days, 1 hour and 33.1230 seconds'

    ```

    Instead, the `minimum_unit` can be changed to have a better resolution;
    the function will still readjust the unit to use the greatest of the
    units that does not lose precision.

    For example setting microseconds but still representing the date with milliseconds:

    ```pycon
    >>> precisedelta(delta, minimum_unit="microseconds")
    '2 days, 1 hour, 33 seconds and 123 milliseconds'

    ```

    If desired, some units can be suppressed: you will not see them represented and the
    time of the other units will be adjusted to keep representing the same timedelta:

    ```pycon
    >>> precisedelta(delta, suppress=['days'])
    '49 hours and 33.12 seconds'

    ```

    Note that microseconds precision is lost if the seconds and all
    the units below are suppressed:

    ```pycon
    >>> delta = dt.timedelta(seconds=90, microseconds=100)
    >>> precisedelta(delta, suppress=['seconds', 'milliseconds', 'microseconds'])
    '1.50 minutes'

    ```

    If the delta is too small to be represented with the minimum unit,
    a value of zero will be returned:

    ```pycon
    >>> delta = dt.timedelta(seconds=1)
    >>> precisedelta(delta, minimum_unit="minutes")
    '0.02 minutes'

    >>> delta = dt.timedelta(seconds=0.1)
    >>> precisedelta(delta, minimum_unit="minutes")
    '0 minutes'

    ```
    """
    date, delta = _date_and_delta(value)
    if date is None:
        return str(value)
    suppress_set = {Unit[s.upper()] for s in suppress}
    min_unit = Unit[minimum_unit.upper()]
    min_unit = _suitable_minimum_unit(min_unit, suppress_set)
    del minimum_unit
    suppress_set = _suppress_lower_units(min_unit, suppress_set)
    days = delta.days
    secs = delta.seconds
    usecs = delta.microseconds
    MICROSECONDS, MILLISECONDS, SECONDS, MINUTES, HOURS, DAYS, MONTHS, YEARS = list(Unit)
    years, days = _quotient_and_remainder(days, 365, YEARS, min_unit, suppress_set)
    months, days = _quotient_and_remainder(days, 30.5, MONTHS, min_unit, suppress_set)
    days, secs = _carry(days, secs, 24 * 3600, DAYS, min_unit, suppress_set)
    hours, secs = _quotient_and_remainder(secs, 3600, HOURS, min_unit, suppress_set)
    minutes, secs = _quotient_and_remainder(secs, 60, MINUTES, min_unit, suppress_set)
    secs, usecs = _carry(secs, usecs, 1000000.0, SECONDS, min_unit, suppress_set)
    msecs, usecs = _quotient_and_remainder(usecs, 1000, MILLISECONDS, min_unit, suppress_set)
    usecs, _unused = _carry(usecs, 0, 1, MICROSECONDS, min_unit, suppress_set)
    fmts = [('%d year', '%d years', years), ('%d month', '%d months', months), ('%d day', '%d days', days), ('%d hour', '%d hours', hours), ('%d minute', '%d minutes', minutes), ('%d second', '%d seconds', secs), ('%d millisecond', '%d milliseconds', msecs), ('%d microsecond', '%d microseconds', usecs)]
    texts: list[str] = []
    for unit, fmt in zip(reversed(Unit), fmts):
        singular_txt, plural_txt, fmt_value = fmt
        if fmt_value > 0 or (not texts and unit == min_unit):
            _fmt_value = 2 if 1 < fmt_value < 2 else int(fmt_value)
            fmt_txt = _ngettext(singular_txt, plural_txt, _fmt_value)
            if unit == min_unit and math.modf(fmt_value)[0] > 0:
                fmt_txt = fmt_txt.replace('%d', format)
            elif unit == YEARS:
                fmt_txt = fmt_txt.replace('%d', '%s')
                texts.append(fmt_txt % intcomma(fmt_value))
                continue
            texts.append(fmt_txt % fmt_value)
        if unit == min_unit:
            break
    if len(texts) == 1:
        return texts[0]
    head = ', '.join(texts[:-1])
    tail = texts[-1]
    return _('%s and %s') % (head, tail)