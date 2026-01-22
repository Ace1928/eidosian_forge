import logging
import os.path
import sys
from .exceptions import NoSuchClassError, UnsupportedPropertyError
from .icon_cache import IconCache
def pyuicMargins(self, widget, prop):
    widget.setContentsMargins(*int_list(prop))