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
def skipIfCircleCI(testfn, reason='Test disabled in CircleCI'):
    """
    Decorate a test to skip if running on CircleCI.
    """
    return unittest.skipIf(is_this_circleci(), reason)(testfn)