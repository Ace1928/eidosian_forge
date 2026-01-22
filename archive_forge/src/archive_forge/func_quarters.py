from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
@property
def quarters(self) -> localedata.LocaleDataDict:
    """Locale display names for quarters.

        >>> Locale('de', 'DE').quarters['format']['wide'][1]
        u'1. Quartal'
        """
    return self._data['quarters']