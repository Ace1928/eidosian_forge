import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def get_modifier_mapping(self):
    """Return a list of eight lists, one for each modifier. The list
        can be indexed using X.ShiftMapIndex, X.Mod1MapIndex, and so on.
        The sublists list the keycodes bound to that modifier."""
    r = request.GetModifierMapping(display=self.display)
    return r.keycodes