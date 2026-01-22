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
def assert_arrays_equal(arrays1, arrays2):
    """Check two iterables yield the same sequence of arrays."""
    for arr1, arr2 in zip_longest(arrays1, arrays2, fillvalue=None):
        assert arr1 is not None and arr2 is not None
        assert_array_equal(arr1, arr2)