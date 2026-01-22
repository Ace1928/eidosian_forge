from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
@property
def default_numbering_system(self) -> str:
    """The default numbering system used by the locale.
        >>> Locale('el', 'GR').default_numbering_system
        u'latn'
        """
    return self._data['default_numbering_system']