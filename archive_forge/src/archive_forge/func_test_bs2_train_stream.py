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
def test_bs2_train_stream(self):
    """
        Test --datatype train:stream.
        """
    return self._run_display_data('train:stream', batchsize=2)