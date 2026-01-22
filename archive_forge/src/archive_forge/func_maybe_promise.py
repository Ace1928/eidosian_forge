from .abstract import Thenable
from .promises import promise
def maybe_promise(p):
    """Return None if p is undefined, otherwise make sure it's a promise."""
    if p:
        if not isinstance(p, Thenable):
            return promise(p)
    return p