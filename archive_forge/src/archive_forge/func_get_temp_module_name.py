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
def get_temp_module_name():
    global _module_num
    get_module_dir()
    name = '_test_ext_module_%d' % _module_num
    _module_num += 1
    if name in sys.modules:
        raise RuntimeError('Temporary module name already in use.')
    return name