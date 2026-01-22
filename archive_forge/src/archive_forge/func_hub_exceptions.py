import os
import sys
import linecache
import re
import inspect
def hub_exceptions(state=True):
    """Toggles whether the hub prints exceptions that are raised from its
    timers.  This can be useful to see how greenthreads are terminating.
    """
    from eventlet import hubs
    hubs.get_hub().set_timer_exceptions(state)
    from eventlet import greenpool
    greenpool.DEBUG = state