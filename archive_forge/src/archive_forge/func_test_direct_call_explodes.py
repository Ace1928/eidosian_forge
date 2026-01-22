import collections
import errno
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from unittest import mock
from oslotest import base as test_base
from oslo_concurrency.fixture import lockutils as fixtures
from oslo_concurrency import lockutils
from oslo_config import fixture as config
def test_direct_call_explodes(self):
    cmd = [sys.executable, '-m', 'oslo_concurrency.lockutils']
    with open(os.devnull, 'w') as devnull:
        retval = subprocess.call(cmd, stderr=devnull)
        self.assertEqual(1, retval)