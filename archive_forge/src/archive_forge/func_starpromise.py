from .abstract import Thenable
from .promises import promise
def starpromise(fun, *args, **kwargs):
    """Create promise, using star arguments."""
    return promise(fun, args, kwargs)