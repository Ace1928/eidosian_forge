import contextlib
import shutil
import tempfile
from pathlib import Path
import numpy
import pytest
from thinc.api import ArgsKwargs, Linear, Padded, Ragged
from thinc.util import has_cupy, is_cupy_array, is_numpy_array
def get_data_checker(inputs):
    if isinstance(inputs, Ragged):
        return assert_raggeds_match
    elif isinstance(inputs, Padded):
        return assert_paddeds_match
    elif isinstance(inputs, list):
        return assert_lists_match
    elif isinstance(inputs, tuple) and len(inputs) == 4:
        return assert_padded_data_match
    elif isinstance(inputs, tuple) and len(inputs) == 2:
        return assert_ragged_data_match
    else:
        return assert_arrays_match