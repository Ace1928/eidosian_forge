import doctest
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock
from unittest.mock import patch
from zope.testing import setupstack
import zc.lockfile
def test_log_formatting(self):
    with patch('os.getpid', Mock(return_value=123)):
        with patch('socket.gethostname', Mock(return_value='myhostname')):
            lock = zc.lockfile.LockFile('f.lock', content_template='{pid}/{hostname}')
            with open('f.lock') as f:
                self.assertEqual(' 123/myhostname\n', f.read())
            lock.close()