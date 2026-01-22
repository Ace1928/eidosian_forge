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
def test_synchronized_wrapped_function_metadata(self):

    @lockutils.synchronized('whatever', 'test-')
    def foo():
        """Bar."""
        pass
    self.assertEqual('Bar.', foo.__doc__, "Wrapped function's docstring got lost")
    self.assertEqual('foo', foo.__name__, "Wrapped function's name got mangled")