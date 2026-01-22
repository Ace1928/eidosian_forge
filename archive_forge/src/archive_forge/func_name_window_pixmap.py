from Xlib import X
from Xlib.protocol import rq
from Xlib.xobject import drawable
def name_window_pixmap(self):
    """Create a new pixmap that refers to the off-screen storage of
    the window, including its border.

    This pixmap will remain allocated until freed whatever happens
    with the window.  However, the window will get a new off-screen
    pixmap every time it is mapped or resized, so to keep track of the
    contents you must listen for these events and get a new pixmap
    after them.
    """
    pid = self.display.allocate_resource_id()
    NameWindowPixmap(display=self.display, opcode=self.display.get_extension_major(extname), window=self, pixmap=pid)
    cls = self.display.get_resource_class('pixmap', drawable.Pixmap)
    return cls(self.display, pid, owner=1)