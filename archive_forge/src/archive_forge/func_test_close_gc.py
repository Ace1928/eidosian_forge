import pytest
import os
import stat
import pickle
import tempfile
import subprocess
import sys
from .common import ut, TestCase, UNICODE_FILENAMES, closed_tempfile
from h5py._hl.files import direct_vfd
from h5py import File
import h5py
from .. import h5
import pathlib
import sys
import h5py
def test_close_gc(writable_file):
    for i in range(100):
        writable_file[str(i)] = []
    filename = writable_file.filename
    writable_file.close()
    for i in range(10):
        with h5py.File(filename, 'r') as f:
            refs = [d.id for d in f.values()]
            refs.append(refs)
            del refs