import logging
import os.path
import sys
from pyside2uic.exceptions import UnsupportedPropertyError
from pyside2uic.icon_cache import IconCache
def pyuicContentsMargins(self, widget, prop):
    widget.setContentsMargins(*int_list(prop))