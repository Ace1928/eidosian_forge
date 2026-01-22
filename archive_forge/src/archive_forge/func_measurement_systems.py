from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
@property
def measurement_systems(self) -> localedata.LocaleDataDict:
    """Localized names for various measurement systems.

        >>> Locale('fr', 'FR').measurement_systems['US']
        u'am\\xe9ricain'
        >>> Locale('en', 'US').measurement_systems['US']
        u'US'

        """
    return self._data['measurement_systems']