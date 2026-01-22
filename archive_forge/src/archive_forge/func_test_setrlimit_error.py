import errno
import io
import logging
import multiprocessing
import os
import pickle
import resource
import socket
import stat
import subprocess
import sys
import tempfile
import time
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_concurrency import processutils
def test_setrlimit_error(self):
    prlimit = self.limit_address_space()
    higher_limit = prlimit.address_space + 1024
    args = [sys.executable, '-m', 'oslo_concurrency.prlimit', '--as=%s' % higher_limit, '--']
    args.extend(self.SIMPLE_PROGRAM)
    try:
        processutils.execute(*args, prlimit=prlimit)
    except processutils.ProcessExecutionError as exc:
        self.assertEqual(1, exc.exit_code)
        self.assertEqual('', exc.stdout)
        expected = '%s -m oslo_concurrency.prlimit: failed to set the AS resource limit: ' % os.path.basename(sys.executable)
        self.assertIn(expected, exc.stderr)
    else:
        self.fail('ProcessExecutionError not raised')