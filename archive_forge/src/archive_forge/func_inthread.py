import warnings
from functools import wraps
from typing import Any, Callable
from twisted.internet import defer, threads
from twisted.internet.defer import Deferred
from scrapy.exceptions import ScrapyDeprecationWarning
def inthread(func: Callable) -> Callable[..., Deferred]:
    """Decorator to call a function in a thread and return a deferred with the
    result
    """

    @wraps(func)
    def wrapped(*a: Any, **kw: Any) -> Deferred:
        return threads.deferToThread(func, *a, **kw)
    return wrapped