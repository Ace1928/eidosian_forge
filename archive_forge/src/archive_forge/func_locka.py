import os
import tempfile
import eventlet
from eventlet import greenpool
from oslotest import base as test_base
from oslo_concurrency import lockutils
def locka(wait):
    a = lockutils.InterProcessLock(os.path.join(tmpdir, 'a'))
    with a:
        wait.wait()
    self.completed = True