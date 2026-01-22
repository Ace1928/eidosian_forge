import re
import string
from Xlib import error
from Xlib.protocol import request
from Xlib.xobject import resource
def store_named_color(self, name, pixel, flags, onerror=None):
    request.StoreNamedColor(display=self.display, onerror=onerror, flags=flags, cmap=self.id, pixel=pixel, name=name)