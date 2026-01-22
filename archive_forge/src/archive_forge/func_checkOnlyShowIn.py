from xdg.IniFile import IniFile, is_ascii
import xdg.Locale
from xdg.Exceptions import ParsingError
from xdg.util import which
import os.path
import re
import warnings
def checkOnlyShowIn(self, value):
    values = self.getList(value)
    valid = ['GNOME', 'KDE', 'LXDE', 'MATE', 'Razor', 'ROX', 'TDE', 'Unity', 'XFCE', 'Old']
    for item in values:
        if item not in valid and item[0:2] != 'X-':
            self.errors.append("'%s' is not a registered OnlyShowIn value" % item)