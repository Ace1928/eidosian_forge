from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def get_currency_name(currency: str, count: float | decimal.Decimal | None=None, locale: Locale | str | None=LC_NUMERIC) -> str:
    """Return the name used by the locale for the specified currency.

    >>> get_currency_name('USD', locale='en_US')
    u'US Dollar'

    .. versionadded:: 0.9.4

    :param currency: the currency code.
    :param count: the optional count.  If provided the currency name
                  will be pluralized to that number if possible.
    :param locale: the `Locale` object or locale identifier.
    """
    loc = Locale.parse(locale)
    if count is not None:
        try:
            plural_form = loc.plural_form(count)
        except (OverflowError, ValueError):
            plural_form = 'other'
        plural_names = loc._data['currency_names_plural']
        if currency in plural_names:
            currency_plural_names = plural_names[currency]
            if plural_form in currency_plural_names:
                return currency_plural_names[plural_form]
            if 'other' in currency_plural_names:
                return currency_plural_names['other']
    return loc.currencies.get(currency, currency)