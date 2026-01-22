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
def skip_if_mvc_enabled(reason):
    """Skip a test if Minor Version Compatibility is enabled"""
    return unittest.skipIf(config.CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY, reason)