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
def generate_only_allocated(self, menu):
    for submenu in menu.Submenus:
        self.generate_only_allocated(submenu)
    if menu.OnlyUnallocated is True:
        self.cache.add_menu_entries(menu.AppDirs)
        menuentries = []
        for rule in menu.Rules:
            menuentries = rule.apply(self.cache.get_menu_entries(menu.AppDirs), 2)
        for menuentry in menuentries:
            if menuentry.Add is True:
                menuentry.Parents.append(menu)
                menu.MenuEntries.append(menuentry)