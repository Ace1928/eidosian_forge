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
def matching_key_for(self, key):
    """
        Given a key, return the actual key stored in self that matches.
        Raise KeyError if the key isn't found.
        """
    try:
        return next((e_key for e_key in self.keys() if e_key == key))
    except StopIteration as err:
        raise KeyError(key) from err