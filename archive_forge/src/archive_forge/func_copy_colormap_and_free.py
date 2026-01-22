import re
import string
from Xlib import error
from Xlib.protocol import request
from Xlib.xobject import resource
def copy_colormap_and_free(self, scr_cmap):
    mid = self.display.allocate_resource_id()
    request.CopyColormapAndFree(display=self.display, mid=mid, src_cmap=src_cmap)
    cls = self.display.get_resource_class('colormap', Colormap)
    return cls(self.display, mid, owner=1)