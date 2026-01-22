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
def ensure_supported_ccs_initialized():
    from numba.cuda import is_available as cuda_is_available
    from numba.cuda.cudadrv import nvvm
    if cuda_is_available():
        nvvm.get_supported_ccs()