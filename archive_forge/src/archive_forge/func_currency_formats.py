from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
@property
def currency_formats(self) -> localedata.LocaleDataDict:
    """Locale patterns for currency number formatting.

        .. note:: The format of the value returned may change between
                  Babel versions.

        >>> Locale('en', 'US').currency_formats['standard']
        <NumberPattern u'\\xa4#,##0.00'>
        >>> Locale('en', 'US').currency_formats['accounting']
        <NumberPattern u'\\xa4#,##0.00;(\\xa4#,##0.00)'>
        """
    return self._data['currency_formats']