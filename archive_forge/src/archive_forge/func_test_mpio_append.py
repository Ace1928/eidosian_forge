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
def test_mpio_append(self, mpi_file_name):
    """ Testing creation of file with append """
    from mpi4py import MPI
    with File(mpi_file_name, 'a', driver='mpio', comm=MPI.COMM_WORLD) as f:
        assert f
        assert f.driver == 'mpio'