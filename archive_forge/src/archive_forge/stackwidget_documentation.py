import sys, traceback
from ..Qt import QtWidgets, QtGui
Display a call stack and exception traceback.

        This allows the user to probe the contents of any frame in the given stack.

        *frame* may either be a Frame instance or None, in which case the current 
        frame is retrieved from ``sys._getframe()``. 

        If *tb* is provided then the frames in the traceback will be appended to 
        the end of the stack list. If *tb* is None, then sys.exc_info() will 
        be checked instead.
        