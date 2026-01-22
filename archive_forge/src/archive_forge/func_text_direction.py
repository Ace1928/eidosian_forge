from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
@property
def text_direction(self) -> str:
    """The text direction for the language in CSS short-hand form.

        >>> Locale('de', 'DE').text_direction
        'ltr'
        >>> Locale('ar', 'SA').text_direction
        'rtl'
        """
    return ''.join((word[0] for word in self.character_order.split('-')))