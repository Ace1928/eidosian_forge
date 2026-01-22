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
def get_fresh_mod(mod_name=__name__):
    my_mod = sys.modules[mod_name]
    try:
        my_mod.__warningregistry__.clear()
    except AttributeError:
        pass
    return my_mod