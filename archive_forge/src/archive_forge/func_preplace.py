from .abstract import Thenable
from .promises import promise
def preplace(p, *args, **kwargs):
    """Replace promise arguments.

    This will force the promise to disregard any arguments
    the promise is fulfilled with, and to be called with the
    provided arguments instead.
    """

    def _replacer(*_, **__):
        return p(*args, **kwargs)
    return promise(_replacer)