import weakref
from inspect import isroutine
from copy import copy
from time import time
from kivy.eventmanager import MODE_DEFAULT_DISPATCH
from kivy.vector import Vector
def dispatch_done(self):
    """Notify that dispatch to the listeners is done.

        Called by the :meth:`EventLoopBase.post_dispatch_input`.

        .. versionadded:: 2.1.0
        """
    self._keep_prev_pos = True
    self._first_dispatch_done = True