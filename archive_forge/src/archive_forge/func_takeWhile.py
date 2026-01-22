from functools import partial, wraps
from typing import Awaitable, Callable, Iterable, Optional, TypeVar
from twisted.internet.defer import Deferred, succeed
def takeWhile(condition: Callable[[_A], bool], xs: Iterable[_A]) -> Iterable[_A]:
    """
    :return: An iterable over C{xs} that stops when C{condition} returns
        ``False`` based on the value of iterated C{xs}.
    """
    for x in xs:
        if condition(x):
            yield x
        else:
            break