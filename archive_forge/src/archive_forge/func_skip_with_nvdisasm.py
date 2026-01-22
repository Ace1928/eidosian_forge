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
def skip_with_nvdisasm(reason):
    nvdisasm_path = shutil.which('nvdisasm')
    return unittest.skipIf(nvdisasm_path is not None, reason)