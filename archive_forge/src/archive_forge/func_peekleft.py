import operator as op
from collections import OrderedDict
from collections.abc import (
from contextlib import contextmanager
from shutil import rmtree
from .core import ENOVAL, Cache
def peekleft(self):
    """Peek at value at front of deque.

        Faster than indexing deque at 0.

        If deque is empty then raise IndexError.

        >>> deque = Deque()
        >>> deque.peekleft()
        Traceback (most recent call last):
            ...
        IndexError: peek from an empty deque
        >>> deque += 'abc'
        >>> deque.peekleft()
        'a'

        :return: value at front of deque
        :raises IndexError: if deque is empty

        """
    default = (None, ENOVAL)
    _, value = self._cache.peek(default=default, side='front', retry=True)
    if value is ENOVAL:
        raise IndexError('peek from an empty deque')
    return value