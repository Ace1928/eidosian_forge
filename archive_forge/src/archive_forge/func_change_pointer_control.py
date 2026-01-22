import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def change_pointer_control(self, accel=None, threshold=None, onerror=None):
    """To change the pointer acceleration, set accel to a tuple (num,
        denum). The pointer will then move num/denum times the normal
        speed if it moves beyond the threshold number of pixels at once.
        To change the threshold, set it to the number of pixels. -1
        restores the default."""
    if accel is None:
        do_accel = 0
        accel_num = 0
        accel_denum = 0
    else:
        do_accel = 1
        accel_num, accel_denum = accel
    if threshold is None:
        do_threshold = 0
    else:
        do_threshold = 1
    request.ChangePointerControl(display=self.display, onerror=onerror, do_accel=do_accel, do_thres=do_threshold, accel_num=accel_num, accel_denum=accel_denum, threshold=threshold)