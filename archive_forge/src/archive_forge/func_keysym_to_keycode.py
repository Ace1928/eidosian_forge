import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def keysym_to_keycode(self, keysym):
    """Look up the primary keycode that is bound to keysym. If
        several keycodes are found, the one with the lowest index and
        lowest code is returned. If keysym is not bound to any key, 0 is
        returned."""
    try:
        return self._keymap_syms[keysym][0][1]
    except (KeyError, IndexError):
        return 0