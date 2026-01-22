import warnings
from functools import wraps
from typing import Any, Callable
from twisted.internet import defer, threads
from twisted.internet.defer import Deferred
from scrapy.exceptions import ScrapyDeprecationWarning
def defers(func: Callable) -> Callable[..., Deferred]:
    """Decorator to make sure a function always returns a deferred"""

    @wraps(func)
    def wrapped(*a: Any, **kw: Any) -> Deferred:
        return defer.maybeDeferred(func, *a, **kw)
    return wrapped