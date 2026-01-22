import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def set_pointer_mapping(self, map):
    """Set the mapping of the pointer buttons. map is a list of
        logical button numbers. map must be of the same length as the
        list returned by Display.get_pointer_mapping().

        map[n] sets the
        logical number for the physical button n+1. Logical number 0
        disables the button. Two physical buttons cannot be mapped to the
        same logical number.

        If one of the buttons to be altered are
        logically in the down state, X.MappingBusy is returned and the
        mapping is not changed. Otherwise the mapping is changed and
        X.MappingSuccess is returned."""
    r = request.SetPointerMapping(display=self.display, map=map)
    return r.status