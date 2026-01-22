from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
@property
def other_numbering_systems(self) -> localedata.LocaleDataDict:
    """
        Mapping of other numbering systems available for the locale.
        See: https://www.unicode.org/reports/tr35/tr35-numbers.html#otherNumberingSystems

        >>> Locale('el', 'GR').other_numbering_systems['traditional']
        u'grek'

        .. note:: The format of the value returned may change between
                  Babel versions.
        """
    return self._data['numbering_systems']