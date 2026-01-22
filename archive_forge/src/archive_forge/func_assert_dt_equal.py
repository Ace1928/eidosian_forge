from __future__ import annotations
import os
import re
import sys
import typing as ty
import unittest
import warnings
from contextlib import nullcontext
from itertools import zip_longest
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from .helpers import assert_data_similar, bytesio_filemap, bytesio_round_trip
from .np_features import memmap_after_ufunc
def assert_dt_equal(a, b):
    """Assert two numpy dtype specifiers are equal

    Avoids failed comparison between int32 / int64 and intp
    """
    assert np.dtype(a).str == np.dtype(b).str