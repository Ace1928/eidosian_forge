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
def mktemp(self, suffix='.hdf5', prefix='', dir=None):
    if dir is None:
        dir = self.tempdir
    return tempfile.mktemp(suffix, prefix, dir=dir)