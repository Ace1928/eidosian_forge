import os
import sys
from .gui import *
from .app_menus import ListedWindow
def find_settings(self):
    if sys.platform == 'darwin':
        home = os.environ['HOME']
        self.setting_file = os.path.join(home, 'Library', 'Preferences', 'org.computop.SnapPy.plist')
    elif sys.platform == 'linux2' or sys.platform == 'linux':
        home = os.environ['HOME']
        self.setting_file = os.path.join(home, '.SnapPy.plist')
    elif sys.platform == 'win32':
        home = os.environ['USERPROFILE']
        self.setting_file = os.path.join(home, '.SnapPy.plist')
    else:
        self.setting_file = None