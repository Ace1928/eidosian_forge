from .abstract import Thenable
from .promises import promise
Wrap promise.

    This wraps the promise such that if the promise is called with a promise as
    argument, we attach ourselves to that promise instead.
    