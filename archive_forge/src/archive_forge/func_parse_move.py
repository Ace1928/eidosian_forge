from xdg.Menu import parse, Menu, MenuEntry
import os
import locale
import subprocess
import ast
import sys
from xdg.BaseDirectory import xdg_data_dirs, xdg_config_dirs
from xdg.DesktopEntry import DesktopEntry
from xdg.Exceptions import ParsingError
from xdg.util import PY3
import xdg.Locale
import xdg.Config
def parse_move(self, node):
    old, new = ('', '')
    for child in node:
        tag, text = (child.tag, child.text)
        text = text.strip() if text else None
        if tag == 'Old' and text:
            old = text
        elif tag == 'New' and text:
            new = text
    return Move(old, new)