from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
@property
def english_name(self) -> str | None:
    """The english display name of the locale.

        >>> Locale('de').english_name
        u'German'
        >>> Locale('de', 'DE').english_name
        u'German (Germany)'

        :type: `unicode`"""
    return self.get_display_name(Locale('en'))