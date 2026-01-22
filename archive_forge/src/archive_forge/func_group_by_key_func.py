import ast
import itertools
import types
from collections import OrderedDict, Counter, defaultdict
from types import FrameType, TracebackType
from typing import (
from asttokens import ASTText
def group_by_key_func(iterable: Iterable[T], key_func: Callable[[T], R]) -> Mapping[R, List[T]]:
    """
    Create a dictionary from an iterable such that the keys are the result of evaluating a key function on elements
    of the iterable and the values are lists of elements all of which correspond to the key.

    >>> def si(d): return sorted(d.items())
    >>> si(group_by_key_func("a bb ccc d ee fff".split(), len))
    [(1, ['a', 'd']), (2, ['bb', 'ee']), (3, ['ccc', 'fff'])]
    >>> si(group_by_key_func([-1, 0, 1, 3, 6, 8, 9, 2], lambda x: x % 2))
    [(0, [0, 6, 8, 2]), (1, [-1, 1, 3, 9])]
    """
    result = defaultdict(list)
    for item in iterable:
        result[key_func(item)].append(item)
    return result