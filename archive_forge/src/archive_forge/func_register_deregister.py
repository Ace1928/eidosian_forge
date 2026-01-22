import collections
import contextlib
import copy
import logging
from oslo_utils import reflection
@contextlib.contextmanager
def register_deregister(notifier, event_type, callback=None, args=None, kwargs=None, details_filter=None):
    """Context manager that registers a callback, then deregisters on exit.

    NOTE(harlowja): if the callback is none, then this registers nothing, which
                    is different from the behavior of the ``register`` method
                    which will *not* accept none as it is not callable...
    """
    if callback is None:
        yield
    else:
        notifier.register(event_type, callback, args=args, kwargs=kwargs, details_filter=details_filter)
        try:
            yield
        finally:
            notifier.deregister(event_type, callback, details_filter=details_filter)