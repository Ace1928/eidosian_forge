from Xlib.protocol import request
from Xlib.xobject import resource
def recolor(self, f_rgb, b_rgb, onerror=None):
    back_red, back_green, back_blue = b_rgb
    fore_red, fore_green, fore_blue = f_rgb
    request.RecolorCursor(display=self.display, onerror=onerror, cursor=self.id, fore_red=fore_red, fore_green=fore_green, fore_blue=fore_blue, back_red=back_red, back_green=back_green, back_blue=back_blue)