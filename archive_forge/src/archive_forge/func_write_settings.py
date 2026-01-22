import os
import sys
from .gui import *
from .app_menus import ListedWindow
def write_settings(self):
    if self.setting_file:
        if hasattr(plistlib, 'dump'):
            with open(self.setting_file, 'wb') as setting_file:
                plistlib.dump(self.setting_dict, setting_file)
        else:
            plistlib.writePlist(self.setting_dict, self.setting_file)