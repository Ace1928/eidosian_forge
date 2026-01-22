from Xlib import X
from Xlib.protocol import rq
from Xlib.xobject import drawable
def redirect_subwindows(self, update):
    """Redirect the hierarchies starting at all current and future
    children to this window to off-screen storage.
    """
    RedirectSubwindows(display=self.display, opcode=self.display.get_extension_major(extname), window=self, update=update)