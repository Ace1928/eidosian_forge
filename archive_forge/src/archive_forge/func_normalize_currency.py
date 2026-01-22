from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def normalize_currency(currency: str, locale: Locale | str | None=None) -> str | None:
    """Returns the normalized identifier of any currency code.

    Accepts a ``locale`` parameter for fined-grained validation, working as
    the one defined above in ``list_currencies()`` method.

    Returns None if the currency is unknown to Babel.
    """
    if isinstance(currency, str):
        currency = currency.upper()
    if not is_currency(currency, locale):
        return None
    return currency