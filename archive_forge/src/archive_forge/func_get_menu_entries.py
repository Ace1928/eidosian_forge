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
def get_menu_entries(self, dirs, legacy=True):
    entries = []
    ids = set()
    appdirs = dirs[:]
    if legacy:
        appdirs.append('legacy')
    key = ''.join(appdirs)
    try:
        return self.cache[key]
    except KeyError:
        pass
    for dir_ in appdirs:
        for menuentry in self.cacheEntries[dir_]:
            try:
                if menuentry.DesktopFileID not in ids:
                    ids.add(menuentry.DesktopFileID)
                    entries.append(menuentry)
                elif menuentry.getType() == MenuEntry.TYPE_SYSTEM:
                    idx = entries.index(menuentry)
                    entry = entries[idx]
                    if entry.getType() == MenuEntry.TYPE_USER:
                        entry.Original = menuentry
            except UnicodeDecodeError:
                continue
    self.cache[key] = entries
    return entries