import sys
import os
from _pydev_bundle._pydev_execfile import execfile
def is_module_ignored(self, modname, modpath):
    for path in [sys.prefix] + self.pathlist:
        if modpath.startswith(path):
            return True
    else:
        return set(modname.split('.')) & set(self.namelist)