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
def unique_strings(strings: Iterator[str], /) -> Iterator[str]:
    """Return iterator over unique strings.

    >>> list(unique_strings(iter(('a', 'b', 'a'))))
    ['a', 'b', 'a2']

    """
    known = set()
    for i, string in enumerate(strings):
        if string in known:
            string += str(i)
        known.add(string)
        yield string