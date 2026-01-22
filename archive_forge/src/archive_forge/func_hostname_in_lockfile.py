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
def hostname_in_lockfile():
    """
    hostname is correctly written into the lock file when it's included in the
    lock file content template

    >>> import zc.lockfile
    >>> with patch('socket.gethostname', Mock(return_value='myhostname')):
    ...     lock = zc.lockfile.LockFile(
    ...         "f.lock", content_template='{hostname}')
    >>> f = open("f.lock")
    >>> _ = f.seek(1)
    >>> f.read().rstrip()
    'myhostname'
    >>> f.close()

    Make sure that locking twice does not overwrite the old hostname:

    >>> lock = zc.lockfile.LockFile("f.lock", content_template='{hostname}')
    Traceback (most recent call last):
      ...
    zc.lockfile.LockError: Couldn't lock 'f.lock'

    >>> f = open("f.lock")
    >>> _ = f.seek(1)
    >>> f.read().rstrip()
    'myhostname'
    >>> f.close()

    >>> lock.close()
    """