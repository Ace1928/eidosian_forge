from __future__ import annotations
import collections.abc
import copy
import functools
import itertools
import operator
import random
import re
from collections.abc import Container, Iterable, Mapping
from typing import Any, Callable, Union
import jaraco.text
def invert_map(map):
    """
    Given a dictionary, return another dictionary with keys and values
    switched. If any of the values resolve to the same key, raises
    a ValueError.

    >>> numbers = dict(a=1, b=2, c=3)
    >>> letters = invert_map(numbers)
    >>> letters[1]
    'a'
    >>> numbers['d'] = 3
    >>> invert_map(numbers)
    Traceback (most recent call last):
    ...
    ValueError: Key conflict in inverted mapping
    """
    res = dict(((v, k) for k, v in map.items()))
    if not len(res) == len(map):
        raise ValueError('Key conflict in inverted mapping')
    return res