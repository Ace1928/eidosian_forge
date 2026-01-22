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
def test_synchronized_without_prefix(self):
    self.config(lock_path=tempfile.mkdtemp(), group='oslo_concurrency')

    @lockutils.synchronized('lock', external=True)
    def test_without_prefix():
        pass
    test_without_prefix()