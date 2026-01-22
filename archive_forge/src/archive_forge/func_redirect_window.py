from Xlib import X
from Xlib.protocol import rq
from Xlib.xobject import drawable
def redirect_window(self, update):
    """Redirect the hierarchy starting at this window to off-screen
    storage.
    """
    RedirectWindow(display=self.display, opcode=self.display.get_extension_major(extname), window=self, update=update)