import functools
import inspect
import eventlet
from eventlet.support import greenlets as greenlet
from eventlet.hubs import get_hub
def wrap_is_timeout(base):
    """Adds `.is_timeout=True` attribute to objects returned by `base()`.

    When `base` is class, attribute is added as read-only property. Returns `base`.
    Otherwise, it returns a function that sets attribute on result of `base()` call.

    Wrappers make best effort to be transparent.
    """
    if inspect.isclass(base):
        base.is_timeout = property(lambda _: True)
        return base

    @functools.wraps(base)
    def fun(*args, **kwargs):
        ex = base(*args, **kwargs)
        ex.is_timeout = True
        return ex
    return fun