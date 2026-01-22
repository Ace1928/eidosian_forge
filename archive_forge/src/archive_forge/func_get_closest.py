import codecs
import csv
import datetime
import gettext
import glob
import os
import re
from tornado import escape
from tornado.log import gen_log
from tornado._locale_data import LOCALE_NAMES
from typing import Iterable, Any, Union, Dict, Optional
@classmethod
def get_closest(cls, *locale_codes: str) -> 'Locale':
    """Returns the closest match for the given locale code."""
    for code in locale_codes:
        if not code:
            continue
        code = code.replace('-', '_')
        parts = code.split('_')
        if len(parts) > 2:
            continue
        elif len(parts) == 2:
            code = parts[0].lower() + '_' + parts[1].upper()
        if code in _supported_locales:
            return cls.get(code)
        if parts[0].lower() in _supported_locales:
            return cls.get(parts[0].lower())
    return cls.get(_default_locale)