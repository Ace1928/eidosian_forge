import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def get_pointer_control(self):
    """Return an object with the following attributes:

        accel_num

        accel_denom
            The acceleration as numerator/denumerator.

        threshold
            The number of pixels the pointer must move before the
            acceleration kicks in."""
    return request.GetPointerControl(display=self.display)