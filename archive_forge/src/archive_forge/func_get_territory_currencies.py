from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def get_territory_currencies(territory: str, start_date: datetime.date | None=None, end_date: datetime.date | None=None, tender: bool=True, non_tender: bool=False, include_details: bool=False) -> list[str] | list[dict[str, Any]]:
    """Returns the list of currencies for the given territory that are valid for
    the given date range.  In addition to that the currency database
    distinguishes between tender and non-tender currencies.  By default only
    tender currencies are returned.

    The return value is a list of all currencies roughly ordered by the time
    of when the currency became active.  The longer the currency is being in
    use the more to the left of the list it will be.

    The start date defaults to today.  If no end date is given it will be the
    same as the start date.  Otherwise a range can be defined.  For instance
    this can be used to find the currencies in use in Austria between 1995 and
    2011:

    >>> from datetime import date
    >>> get_territory_currencies('AT', date(1995, 1, 1), date(2011, 1, 1))
    ['ATS', 'EUR']

    Likewise it's also possible to find all the currencies in use on a
    single date:

    >>> get_territory_currencies('AT', date(1995, 1, 1))
    ['ATS']
    >>> get_territory_currencies('AT', date(2011, 1, 1))
    ['EUR']

    By default the return value only includes tender currencies.  This
    however can be changed:

    >>> get_territory_currencies('US')
    ['USD']
    >>> get_territory_currencies('US', tender=False, non_tender=True,
    ...                          start_date=date(2014, 1, 1))
    ['USN', 'USS']

    .. versionadded:: 2.0

    :param territory: the name of the territory to find the currency for.
    :param start_date: the start date.  If not given today is assumed.
    :param end_date: the end date.  If not given the start date is assumed.
    :param tender: controls whether tender currencies should be included.
    :param non_tender: controls whether non-tender currencies should be
                       included.
    :param include_details: if set to `True`, instead of returning currency
                            codes the return value will be dictionaries
                            with detail information.  In that case each
                            dictionary will have the keys ``'currency'``,
                            ``'from'``, ``'to'``, and ``'tender'``.
    """
    currencies = get_global('territory_currencies')
    if start_date is None:
        start_date = datetime.date.today()
    elif isinstance(start_date, datetime.datetime):
        start_date = start_date.date()
    if end_date is None:
        end_date = start_date
    elif isinstance(end_date, datetime.datetime):
        end_date = end_date.date()
    curs = currencies.get(territory.upper(), ())

    def _is_active(start, end):
        return (start is None or start <= end_date) and (end is None or end >= start_date)
    result = []
    for currency_code, start, end, is_tender in curs:
        if start:
            start = datetime.date(*start)
        if end:
            end = datetime.date(*end)
        if (is_tender and tender or (not is_tender and non_tender)) and _is_active(start, end):
            if include_details:
                result.append({'currency': currency_code, 'from': start, 'to': end, 'tender': is_tender})
            else:
                result.append(currency_code)
    return result