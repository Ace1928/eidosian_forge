import logging
import os.path
import sys
from .exceptions import NoSuchClassError, UnsupportedPropertyError
from .icon_cache import IconCache
def qfont_enum(v):
    return getattr(QtGui.QFont, v)