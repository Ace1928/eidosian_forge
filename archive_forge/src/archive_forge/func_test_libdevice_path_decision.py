import sys
import os
import multiprocessing as mp
import warnings
from numba.core.config import IS_WIN32, IS_OSX
from numba.core.errors import NumbaWarning
from numba.cuda.cudadrv import nvvm
from numba.cuda.testing import (
from numba.cuda.cuda_paths import (
def test_libdevice_path_decision(self):
    by, info, warns = self.remote_do(self.do_clear_envs)
    if has_cuda:
        self.assertEqual(by, 'Conda environment')
    else:
        self.assertEqual(by, '<unknown>')
        self.assertIsNone(info)
    self.assertFalse(warns)
    by, info, warns = self.remote_do(self.do_set_cuda_home)
    self.assertEqual(by, 'CUDA_HOME')
    self.assertEqual(info, os.path.join('mycudahome', 'nvvm', 'libdevice'))
    self.assertFalse(warns)
    if get_system_ctk() is None:
        by, info, warns = self.remote_do(self.do_clear_envs)
        self.assertEqual(by, '<unknown>')
        self.assertIsNone(info)
        self.assertFalse(warns)
    else:
        by, info, warns = self.remote_do(self.do_clear_envs)
        self.assertEqual(by, 'System')
        self.assertFalse(warns)