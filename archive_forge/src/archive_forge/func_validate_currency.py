from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def validate_currency(currency: str, locale: Locale | str | None=None) -> None:
    """ Check the currency code is recognized by Babel.

    Accepts a ``locale`` parameter for fined-grained validation, working as
    the one defined above in ``list_currencies()`` method.

    Raises a `UnknownCurrencyError` exception if the currency is unknown to Babel.
    """
    if currency not in list_currencies(locale):
        raise UnknownCurrencyError(currency)