import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def keysym_to_keycodes(self, keysym):
    """Look up all the keycodes that is bound to keysym. A list of
        tuples (keycode, index) is returned, sorted primarily on the
        lowest index and secondarily on the lowest keycode."""
    try:
        return [(x[1], x[0]) for x in self._keymap_syms[keysym]]
    except KeyError:
        return []