import sys
import os
import shutil
import inspect
import tempfile
import subprocess
from contextlib import contextmanager
from functools import wraps
import numpy as np
from numpy.lib.recfunctions import repack_fields
import h5py
import unittest as ut
@contextmanager
def closed_tempfile(suffix='', text=None):
    """
    Context manager which yields the path to a closed temporary file with the
    suffix `suffix`. The file will be deleted on exiting the context. An
    additional argument `text` can be provided to have the file contain `text`.
    """
    with tempfile.NamedTemporaryFile('w+t', suffix=suffix, delete=False) as test_file:
        file_name = test_file.name
        if text is not None:
            test_file.write(text)
            test_file.flush()
    yield file_name
    shutil.rmtree(file_name, ignore_errors=True)