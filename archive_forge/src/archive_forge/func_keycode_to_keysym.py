import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def keycode_to_keysym(self, keycode, index):
    """Convert a keycode to a keysym, looking in entry index.
        Normally index 0 is unshifted, 1 is shifted, 2 is alt grid, and 3
        is shift+alt grid. If that key entry is not bound, X.NoSymbol is
        returned."""
    try:
        return self._keymap_codes[keycode][index]
    except IndexError:
        return X.NoSymbol