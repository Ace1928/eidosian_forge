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
def subproc_env(d):
    """Set environment variables for the @insubprocess decorator"""

    def decorator(f):
        f.subproc_env = d
        return f
    return decorator