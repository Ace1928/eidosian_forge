from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
@property
def ordinal_form(self) -> PluralRule:
    """Plural rules for the locale.

        >>> Locale('en').ordinal_form(1)
        'one'
        >>> Locale('en').ordinal_form(2)
        'two'
        >>> Locale('en').ordinal_form(3)
        'few'
        >>> Locale('fr').ordinal_form(2)
        'other'
        >>> Locale('ru').ordinal_form(100)
        'other'
        """
    return self._data.get('ordinal_form', _default_plural_rule)