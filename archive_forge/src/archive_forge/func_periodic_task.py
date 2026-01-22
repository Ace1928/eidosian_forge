import copy
import logging
import random
import time
from time import monotonic as now
from oslo_service._i18n import _
from oslo_service import _options
from oslo_utils import reflection
def periodic_task(*args, **kwargs):
    """Decorator to indicate that a method is a periodic task.

    This decorator can be used in two ways:

        1. Without arguments '@periodic_task', this will be run on the default
           interval of 60 seconds.

        2. With arguments:
           @periodic_task(spacing=N [, run_immediately=[True|False]]
           [, name=[None|"string"])
           this will be run on approximately every N seconds. If this number is
           negative the periodic task will be disabled. If the run_immediately
           argument is provided and has a value of 'True', the first run of the
           task will be shortly after task scheduler starts.  If
           run_immediately is omitted or set to 'False', the first time the
           task runs will be approximately N seconds after the task scheduler
           starts. If name is not provided, __name__ of function is used.
    """

    def decorator(f):
        if 'ticks_between_runs' in kwargs:
            raise InvalidPeriodicTaskArg(arg='ticks_between_runs')
        f._periodic_task = True
        f._periodic_external_ok = kwargs.pop('external_process_ok', False)
        f._periodic_enabled = kwargs.pop('enabled', True)
        f._periodic_name = kwargs.pop('name', f.__name__)
        f._periodic_spacing = kwargs.pop('spacing', 0)
        f._periodic_immediate = kwargs.pop('run_immediately', False)
        if f._periodic_immediate:
            f._periodic_last_run = None
        else:
            f._periodic_last_run = now()
        return f
    if kwargs:
        return decorator
    else:
        return decorator(args[0])