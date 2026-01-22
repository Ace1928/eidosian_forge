from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
@property
def datetime_formats(self) -> localedata.LocaleDataDict:
    """Locale patterns for datetime formatting.

        .. note:: The format of the value returned may change between
                  Babel versions.

        >>> Locale('en').datetime_formats['full']
        u'{1}, {0}'
        >>> Locale('th').datetime_formats['medium']
        u'{1} {0}'
        """
    return self._data['datetime_formats']