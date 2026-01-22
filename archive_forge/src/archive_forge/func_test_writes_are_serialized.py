import os
import sys
import threading
from io import BytesIO
from textwrap import dedent
import configobj
from testtools import matchers
from .. import (bedding, branch, config, controldir, diff, errors, lock,
from .. import registry as _mod_registry
from .. import tests, trace
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..bzr import remote
from ..transport import remote as transport_remote
from . import features, scenarios, test_server
def test_writes_are_serialized(self):
    c1 = self.stack
    c2 = self.get_stack(self)
    before_writing = threading.Event()
    after_writing = threading.Event()
    writing_done = threading.Event()
    c1_save_without_locking_orig = c1.store.save_without_locking

    def c1_save_without_locking():
        before_writing.set()
        c1_save_without_locking_orig()
        after_writing.wait()
    c1.store.save_without_locking = c1_save_without_locking

    def c1_set():
        c1.set('one', 'c1')
        writing_done.set()
    t1 = threading.Thread(target=c1_set)
    self.addCleanup(t1.join)
    self.addCleanup(after_writing.set)
    t1.start()
    before_writing.wait()
    self.assertRaises(errors.LockContention, c2.set, 'one', 'c2')
    self.assertEqual('c1', c1.get('one'))
    after_writing.set()
    writing_done.wait()
    c2.set('one', 'c2')
    self.assertEqual('c2', c2.get('one'))