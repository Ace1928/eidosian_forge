from __future__ import annotations
import functools
import itertools
import os.path
import re
import textwrap
from email.message import Message
from email.parser import Parser
from typing import Iterator
from .vendored.packaging.requirements import Requirement
@functools.singledispatch
def yield_lines(iterable):
    """
    Yield valid lines of a string or iterable.
    >>> list(yield_lines(''))
    []
    >>> list(yield_lines(['foo', 'bar']))
    ['foo', 'bar']
    >>> list(yield_lines('foo\\nbar'))
    ['foo', 'bar']
    >>> list(yield_lines('\\nfoo\\n#bar\\nbaz #comment'))
    ['foo', 'baz #comment']
    >>> list(yield_lines(['foo\\nbar', 'baz', 'bing\\n\\n\\n']))
    ['foo', 'bar', 'baz', 'bing']
    """
    return itertools.chain.from_iterable(map(yield_lines, iterable))