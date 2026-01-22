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
def epics_datetime(sec: int, nsec: int, /) -> datetime.datetime:
    """Return datetime object from epicsTSSec and epicsTSNsec tag values.

    >>> epics_datetime(802117916, 103746502)
    datetime.datetime(2015, 6, 2, 11, 31, 56, 103746)

    """
    return datetime.datetime.fromtimestamp(sec + 631152000 + nsec / 1000000000.0)