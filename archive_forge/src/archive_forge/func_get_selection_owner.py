import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def get_selection_owner(self, selection):
    """Return the window that owns selection (an atom), or X.NONE if
        there is no owner for the selection. Can raise BadAtom."""
    r = request.GetSelectionOwner(display=self.display, selection=selection)
    return r.owner