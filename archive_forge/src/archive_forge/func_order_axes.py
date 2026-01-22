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
def order_axes(indices: ArrayLike, /, squeeze: bool=False) -> tuple[int, ...]:
    """Return order of axes sorted by variations in indices.

    Parameters:
        indices:
            Multi-dimensional indices of chunks in array.
        squeeze:
            Remove length-1 dimensions of nonvarying axes.

    Returns:
        Order of axes sorted by variations in indices.
        The axis with the least variations in indices is returned first,
        the axis varying fastest is last.

    Examples:
        First axis varies fastest, second axis is squeezed:
        >>> order_axes([(0, 2, 0), (1, 2, 0), (0, 2, 1), (1, 2, 1)], True)
        (2, 0)

    """
    diff = numpy.sum(numpy.abs(numpy.diff(indices, axis=0)), axis=0).tolist()
    order = tuple(sorted(range(len(diff)), key=diff.__getitem__))
    if squeeze:
        order = tuple((i for i in order if diff[i] != 0))
    return order