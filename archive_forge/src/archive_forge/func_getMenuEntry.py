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
def getMenuEntry(self, desktopfileid, deep=False):
    """Searches for a MenuEntry with a given DesktopFileID."""
    for menuentry in self.MenuEntries:
        if menuentry.DesktopFileID == desktopfileid:
            return menuentry
    if deep:
        for submenu in self.Submenus:
            submenu.getMenuEntry(desktopfileid, deep)