import sys
import types
import eventlet
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from heat.common.i18n import _
from heat.common import timeutils
def task_description(task):
    """Return a human-readable string description of a task.

    The description is used to identify the task when logging its status.
    """
    name = task.__name__ if hasattr(task, '__name__') else None
    if name is not None and isinstance(task, (types.MethodType, types.FunctionType)):
        if getattr(task, '__self__', None) is not None:
            return '%s from %s' % (str(name), task.__self__)
        else:
            return str(name)
    return encodeutils.safe_decode(repr(task))