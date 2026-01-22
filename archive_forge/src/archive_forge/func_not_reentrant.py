from fixtures import Fixture
import signal
from typing import Union
from ._deferreddebug import DebugTwisted
from twisted.internet import defer
from twisted.internet.interfaces import IReactorThreads
from twisted.python.failure import Failure
from twisted.python.util import mergeFunctionMetadata
def not_reentrant(function, _calls={}):
    """Decorates a function as not being re-entrant.

    The decorated function will raise an error if called from within itself.
    """

    def decorated(*args, **kwargs):
        if _calls.get(function, False):
            raise ReentryError(function)
        _calls[function] = True
        try:
            return function(*args, **kwargs)
        finally:
            _calls[function] = False
    return mergeFunctionMetadata(function, decorated)