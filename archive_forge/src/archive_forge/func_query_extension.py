import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def query_extension(self, name):
    """Ask the server if it supports the extension name. If it is
        supported an object with the following attributes is returned:

        major_opcode
            The major opcode that the requests of this extension uses.
        first_event
            The base event code if the extension have additional events, or 0.
        first_error
            The base error code if the extension have additional errors, or 0.

        If the extension is not supported, None is returned."""
    r = request.QueryExtension(display=self.display, name=name)
    if r.present:
        return r
    else:
        return None