from __future__ import annotations
import gettext
import importlib
import json
import locale
import os
import re
import sys
import traceback
from functools import lru_cache
from typing import Any, Pattern
import babel
from packaging.version import parse as parse_version
def update_locale(self, locale_: str) -> None:
    """
        Update the locale.

        Parameters
        ----------
        locale_: str
            The language name to use.
        """
    self._locale = locale_
    localedir = None
    if locale_ != DEFAULT_LOCALE:
        language_pack_module = f'jupyterlab_language_pack_{locale_}'
        try:
            mod = importlib.import_module(language_pack_module)
            assert mod.__file__ is not None
            localedir = os.path.join(os.path.dirname(mod.__file__), LOCALE_DIR)
        except Exception:
            pass
    self._translator = gettext.translation(self._domain, localedir=localedir, languages=(self._locale,), fallback=True)