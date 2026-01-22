import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def set_modifier_mapping(self, keycodes):
    """Set the keycodes for the eight modifiers X.Shift, X.Lock,
        X.Control, X.Mod1, X.Mod2, X.Mod3, X.Mod4 and X.Mod5. keycodes
        should be a eight-element list where each entry is a list of the
        keycodes that should be bound to that modifier.

        If any changed
        key is logically in the down state, X.MappingBusy is returned and
        the mapping is not changed. If the mapping violates some server
        restriction, X.MappingFailed is returned. Otherwise the mapping
        is changed and X.MappingSuccess is returned."""
    r = request.SetModifierMapping(display=self.display, keycodes=keycodes)
    return r.status