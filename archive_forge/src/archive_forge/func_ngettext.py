import gettext
import locale
import os.path
import sys
from typing import Optional, cast, List
from .. import package_dir
def ngettext(singular, plural, n):
    return translator.ngettext(singular, plural, n)