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
def friendly_number(self, value: int) -> str:
    """Returns a comma-separated number for the given integer."""
    if self.code not in ('en', 'en_US'):
        return str(value)
    s = str(value)
    parts = []
    while s:
        parts.append(s[-3:])
        s = s[:-3]
    return ','.join(reversed(parts))