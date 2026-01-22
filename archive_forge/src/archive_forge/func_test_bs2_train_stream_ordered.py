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
def test_bs2_train_stream_ordered(self):
    """
        Test --datatype train:stream:ordered.
        """
    return self._run_display_data('train:stream:ordered', batchsize=2)