import contextlib
import shutil
import tempfile
from pathlib import Path
import numpy
import pytest
from thinc.api import ArgsKwargs, Linear, Padded, Ragged
from thinc.util import has_cupy, is_cupy_array, is_numpy_array
def assert_ragged_data_match(X, Y):
    return assert_raggeds_match(Ragged(*X), Ragged(*Y))