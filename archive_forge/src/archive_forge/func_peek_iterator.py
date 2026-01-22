from __future__ import annotations
import binascii
import collections
import datetime
import enum
import glob
import io
import json
import logging
import math
import os
import re
import struct
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
import numpy
from typing import TYPE_CHECKING, BinaryIO, cast, final, overload
def peek_iterator(iterator: Iterator[Any], /) -> tuple[Any, Iterator[Any]]:
    """Return first item of iterator and iterator.

    >>> first, it = peek_iterator(iter((0, 1, 2)))
    >>> first
    0
    >>> list(it)
    [0, 1, 2]

    """
    first = next(iterator)

    def newiter(first: Any=first, iterator: Iterator[Any]=iterator) -> Iterator[Any]:
        yield first
        yield from iterator
    return (first, newiter())