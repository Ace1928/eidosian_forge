from importlib.util import spec_from_file_location, module_from_spec
import os
import pathlib
import pytest
import shutil
import subprocess
import sys
import sysconfig
import textwrap
import warnings
import numpy as np
from numpy.testing import IS_WASM
@pytest.mark.skipif(numba is None or cffi is None, reason='requires numba and cffi')
def test_numba():
    from numpy.random._examples.numba import extending