from __future__ import annotations
import re
import typing
from collections.abc import Iterable, Sequence
from contextlib import suppress
from copy import deepcopy
from dataclasses import fields
from typing import Any, Dict
import pandas as pd
from ..iapi import labels_view
from .evaluation import after_stat, stage
def strip_stat(value):
    """
    Remove stat function that mark calculated aesthetics

    Parameters
    ----------
    value : object
        Aesthetic value. In most cases this will be a string
        but other types will pass through unmodified.

    Return
    ------
    out : object
        Aesthetic value with the dots removed.

    >>> strip_stat("stat(density + stat(count))")
    density + count

    >>> strip_stat("stat(density) + 5")
    density + 5

    >>> strip_stat("5 + stat(func(density))")
    5 + func(density)

    >>> strip_stat("stat(func(density) + var1)")
    func(density) + var1

    >>> strip_stat("stat + var1")
    stat + var1

    >>> strip_stat(4)
    4
    """

    def strip_hanging_closing_parens(s):
        """
        Remove leftover  parens
        """
        stack = 0
        idx = []
        for i, c in enumerate(s):
            if c == '(':
                stack += 1
            elif c == ')':
                stack -= 1
                if stack < 0:
                    idx.append(i)
                    stack = 0
                    continue
            yield c
    with suppress(TypeError):
        if STAT_RE.search(value):
            value = re.sub('\\bstat\\(', '', value)
            value = ''.join(strip_hanging_closing_parens(value))
    return value