import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def create_resource_object(self, type, id):
    """Create a resource object of type for the integer id. type
        should be one of the following strings:

        resource
        drawable
        window
        pixmap
        fontable
        font
        gc
        colormap
        cursor

        This function can be used when a resource ID has been fetched
        e.g. from an resource or a command line argument. Resource
        objects should never be created by instantiating the appropriate
        class directly, since any X extensions dynamically added by the
        library will not be available.
        """
    return self.display.resource_classes[type](self.display, id)