import os
import platform
import shutil
from numba.tests.support import SerialMixin
from numba.cuda.cuda_paths import get_conda_ctk
from numba.cuda.cudadrv import driver, devices, libs
from numba.core import config
from numba.tests.support import TestCase
from pathlib import Path
import unittest
def skip_if_mvc_libraries_unavailable(fn):
    libs_available = False
    try:
        import cubinlinker
        import ptxcompiler
        libs_available = True
    except ImportError:
        pass
    return unittest.skipUnless(libs_available, 'Requires cubinlinker and ptxcompiler')(fn)