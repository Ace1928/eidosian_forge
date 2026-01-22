from __future__ import annotations
import datetime
import inspect
import logging
import re
import sys
import typing as t
from collections.abc import Collection, Set
from contextlib import contextmanager
from copy import copy
from enum import Enum
from itertools import count
def name_sequence(prefix: str) -> t.Callable[[], str]:
    """Returns a name generator given a prefix (e.g. a0, a1, a2, ... if the prefix is "a")."""
    sequence = count()
    return lambda: f'{prefix}{next(sequence)}'