import sys
import os
import time
import locale
import signal
import urwid
from typing import Optional
from . import args as bpargs, repl, translations
from .formatter import theme_map
from .translations import _
from .keys import urwid_key_dispatch as key_dispatch
def decoding_input_filter(keys, raw):
    """Input filter for urwid which decodes each key with the locale's
    preferred encoding.'"""
    encoding = locale.getpreferredencoding()
    converted_keys = list()
    for key in keys:
        if isinstance(key, str):
            converted_keys.append(key.decode(encoding))
        else:
            converted_keys.append(key)
    return converted_keys