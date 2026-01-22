from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
@property
def list_patterns(self) -> localedata.LocaleDataDict:
    """Patterns for generating lists

        .. note:: The format of the value returned may change between
                  Babel versions.

        >>> Locale('en').list_patterns['standard']['start']
        u'{0}, {1}'
        >>> Locale('en').list_patterns['standard']['end']
        u'{0}, and {1}'
        >>> Locale('en_GB').list_patterns['standard']['end']
        u'{0} and {1}'
        """
    return self._data['list_patterns']