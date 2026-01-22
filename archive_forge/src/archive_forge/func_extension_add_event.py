import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def extension_add_event(self, code, evt, name=None):
    """extension_add_event(code, evt, [name])

        Add an extension event.  CODE is the numeric code, and EVT is
        the event class.  EVT will be cloned, and the attribute _code
        of the new event class will be set to CODE.

        If NAME is omitted, it will be set to the name of EVT.  This
        name is used to insert an entry in the DictWrapper
        extension_event.
        """
    newevt = type(evt.__name__, evt.__bases__, evt.__dict__.copy())
    newevt._code = code
    self.display.add_extension_event(code, newevt)
    if name is None:
        name = evt.__name__
    setattr(self.extension_event, name, code)