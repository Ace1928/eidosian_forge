import concurrent.futures
import contextvars
import logging
import sys
from types import GenericAlias
from . import base_futures
from . import events
from . import exceptions
from . import format_helpers
def wrap_future(future, *, loop=None):
    """Wrap concurrent.futures.Future object."""
    if isfuture(future):
        return future
    assert isinstance(future, concurrent.futures.Future), f'concurrent.futures.Future is expected, got {future!r}'
    if loop is None:
        loop = events._get_event_loop()
    new_future = loop.create_future()
    _chain_future(future, new_future)
    return new_future