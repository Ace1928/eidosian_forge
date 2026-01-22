from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
@property
def unit_display_names(self) -> localedata.LocaleDataDict:
    """Display names for units of measurement.

        .. seealso::

           You may want to use :py:func:`babel.units.get_unit_name` instead.

        .. note:: The format of the value returned may change between
                  Babel versions.

        """
    return self._data['unit_display_names']