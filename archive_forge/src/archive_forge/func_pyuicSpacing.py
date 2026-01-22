import logging
import os.path
import sys
from .exceptions import NoSuchClassError, UnsupportedPropertyError
from .icon_cache import IconCache
def pyuicSpacing(self, widget, prop):
    horiz, vert = int_list(prop)
    if horiz == vert:
        widget.setSpacing(horiz)
    else:
        if horiz >= 0:
            widget.setHorizontalSpacing(horiz)
        if vert >= 0:
            widget.setVerticalSpacing(vert)