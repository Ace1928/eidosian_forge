from Xlib import X
from Xlib.protocol import rq
from Xlib.xobject import drawable
def unredirect_window(self, update):
    """Stop redirecting this window hierarchy.
    """
    UnredirectWindow(display=self.display, opcode=self.display.get_extension_major(extname), window=self, update=update)