from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
@property
def meta_zones(self) -> localedata.LocaleDataDict:
    """Locale display names for meta time zones.

        Meta time zones are basically groups of different Olson time zones that
        have the same GMT offset and daylight savings time.

        .. note:: The format of the value returned may change between
                  Babel versions.

        >>> Locale('en', 'US').meta_zones['Europe_Central']['long']['daylight']
        u'Central European Summer Time'

        .. versionadded:: 0.9
        """
    return self._data['meta_zones']