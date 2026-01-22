from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
@property
def zone_formats(self) -> localedata.LocaleDataDict:
    """Patterns related to the formatting of time zones.

        .. note:: The format of the value returned may change between
                  Babel versions.

        >>> Locale('en', 'US').zone_formats['fallback']
        u'%(1)s (%(0)s)'
        >>> Locale('pt', 'BR').zone_formats['region']
        u'Hor\\xe1rio %s'

        .. versionadded:: 0.9
        """
    return self._data['zone_formats']