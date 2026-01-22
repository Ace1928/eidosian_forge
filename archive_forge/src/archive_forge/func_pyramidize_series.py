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
def pyramidize_series(series: list[TiffPageSeries], /, isreduced: bool=False) -> None:
    """Pyramidize list of TiffPageSeries in-place.

    TiffPageSeries that are a subresolution of another TiffPageSeries are
    appended to the other's TiffPageSeries levels and removed from the list.
    Levels are to be ordered by size using the same downsampling factor.
    TiffPageSeries of subifds cannot be pyramid top levels.

    """
    samplingfactors = (2, 3, 4)
    i = 0
    while i < len(series):
        a = series[i]
        p = None
        j = i + 1
        if a.keyframe.is_subifd:
            i += 1
            continue
        while j < len(series):
            b = series[j]
            if isreduced and (not b.keyframe.is_reduced):
                j += 1
                continue
            if p is None:
                for f in samplingfactors:
                    if subresolution(a.levels[-1], b, p=f) == 1:
                        p = f
                        break
                else:
                    j += 1
                    continue
            elif subresolution(a.levels[-1], b, p=p) != 1:
                j += 1
                continue
            a.levels.append(b)
            del series[j]
        i += 1