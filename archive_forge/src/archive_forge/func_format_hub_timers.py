import os
import sys
import linecache
import re
import inspect
def format_hub_timers():
    """ Returns a formatted string of the current timers on the current
    hub.  This can be useful in determining what's going on in the event system,
    especially when used in conjunction with :func:`hub_timer_stacks`.
    """
    from eventlet import hubs
    hub = hubs.get_hub()
    result = ['TIMERS:']
    for l in hub.timers:
        result.append(repr(l))
    return os.linesep.join(result)