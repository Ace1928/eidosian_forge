import os
import unittest
import contextlib
import tempfile
import shutil
import io
import signal
from typing import Tuple, Dict, Any
from parlai.core.opt import Opt
import parlai.utils.logging as logging
def skipUnlessGPU(testfn, reason='Test requires a GPU'):
    """
    Decorate a test to skip if no GPU is available.
    """
    return unittest.skipUnless(GPU_AVAILABLE, reason)(testfn)