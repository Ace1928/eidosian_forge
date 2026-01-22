import glob
import os
import sys
import subprocess
import tempfile
import shutil
import atexit
import textwrap
import re
import pytest
import contextlib
import numpy
from pathlib import Path
from numpy.compat import asstr
from numpy._utils import asunicode
from numpy.testing import temppath, IS_WASM
from importlib import import_module
import os
import sys
@contextlib.contextmanager
def switchdir(path):
    curpath = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(curpath)