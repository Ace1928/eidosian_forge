import sys
import os
import webbrowser
from urllib.request import pathname2url
from .gui import *
from . import __file__ as snappy_dir
from .infowindow import about_snappy, InfoWindow
from .version import version
import shutil
def build_entries(self):
    if sys.platform == 'darwin':
        return
    self.delete(0, self.index(Tk_.END))
    for object in self.window_list:
        self.add_command(label=object.menu_title, command=object.bring_to_front)