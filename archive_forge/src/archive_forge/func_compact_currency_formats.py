from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
@property
def compact_currency_formats(self) -> localedata.LocaleDataDict:
    """Locale patterns for compact currency number formatting.

        .. note:: The format of the value returned may change between
                  Babel versions.

        >>> Locale('en', 'US').compact_currency_formats["short"]["one"]["1000"]
        <NumberPattern u'¤0K'>
        """
    return self._data['compact_currency_formats']