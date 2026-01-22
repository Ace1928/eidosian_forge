from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def get_currency_unit_pattern(currency: str, count: float | decimal.Decimal | None=None, locale: Locale | str | None=LC_NUMERIC) -> str:
    """
    Return the unit pattern used for long display of a currency value
    for a given locale.
    This is a string containing ``{0}`` where the numeric part
    should be substituted and ``{1}`` where the currency long display
    name should be substituted.

    >>> get_currency_unit_pattern('USD', locale='en_US', count=10)
    u'{0} {1}'

    .. versionadded:: 2.7.0

    :param currency: the currency code.
    :param count: the optional count.  If provided the unit
                  pattern for that number will be returned.
    :param locale: the `Locale` object or locale identifier.
    """
    loc = Locale.parse(locale)
    if count is not None:
        plural_form = loc.plural_form(count)
        try:
            return loc._data['currency_unit_patterns'][plural_form]
        except LookupError:
            pass
    return loc._data['currency_unit_patterns']['other']