import contextlib
import io
import logging
import os
import pwd
import shutil
import signal
import sys
import threading
import time
from unittest import mock
import fixtures
import testtools
from testtools import content
from oslo_rootwrap import client
from oslo_rootwrap import cmd
from oslo_rootwrap import subprocess
from oslo_rootwrap.tests import run_daemon
def test_graceful_death(self):
    tmpdir = self.useFixture(fixtures.TempDir()).path
    fifo_path = os.path.join(tmpdir, 'fifo')
    os.mkfifo(fifo_path)
    self.execute(['cat'])
    t = threading.Thread(target=self._exec_thread, args=(fifo_path,))
    t.start()
    with open(fifo_path) as f:
        f.readline()
    os.kill(self.client._process.pid, signal.SIGTERM)
    t.join()
    if isinstance(self._thread_res, Exception):
        raise self._thread_res
    code, out, err = self._thread_res
    self.assertEqual(0, code)
    self.assertEqual('OK\n', out)
    self.assertEqual('', err)