import re
import string
from Xlib import error
from Xlib.protocol import request
from Xlib.xobject import resource
def uninstall_colormap(self, onerror=None):
    request.UninstallColormap(display=self.display, onerror=onerror, cmap=self.id)