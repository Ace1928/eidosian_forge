import codecs
import errno
import os
import re
import stat
import sys
import time
from functools import partial
from typing import Dict, List
from .lazy_import import lazy_import
import locale
import ntpath
import posixpath
import select
import shutil
from shutil import rmtree
import socket
import subprocess
import unicodedata
from breezy import (
from breezy.i18n import gettext
from hashlib import md5
from hashlib import sha1 as sha
import breezy
from . import errors
def format_local_date(t, offset=0, timezone='original', date_fmt=None, show_offset=True):
    """Return an unicode date string formatted according to the current locale.

    :param t: Seconds since the epoch.
    :param offset: Timezone offset in seconds east of utc.
    :param timezone: How to display the time: 'utc', 'original' for the
         timezone specified by offset, or 'local' for the process's current
         timezone.
    :param date_fmt: strftime format.
    :param show_offset: Whether to append the timezone.
    """
    date_fmt, tt, offset_str = _format_date(t, offset, timezone, date_fmt, show_offset)
    date_str = time.strftime(date_fmt, tt)
    if not isinstance(date_str, str):
        date_str = date_str.decode(get_user_encoding(), 'replace')
    return date_str + offset_str