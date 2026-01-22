import os
import sys
from .gui import *
from .app_menus import ListedWindow
def current_font_tuple(self):
    font = self.current_font_dict()
    style = '%s %s' % (font['weight'], font['slant'])
    return (font['family'], font['size'], style)