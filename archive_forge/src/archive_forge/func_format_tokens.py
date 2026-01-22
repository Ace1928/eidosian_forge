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
def format_tokens(tokensource):
    for token, text in tokensource:
        if text == '\n':
            continue
        while token not in theme_map:
            token = token.parent
        yield (theme_map[token], text)