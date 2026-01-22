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
def test_read_while_writing(self):
    c1 = self.stack
    ready_to_write = threading.Event()
    do_writing = threading.Event()
    writing_done = threading.Event()
    c1_save_without_locking_orig = c1.store.save_without_locking

    def c1_save_without_locking():
        ready_to_write.set()
        do_writing.wait()
        c1_save_without_locking_orig()
        writing_done.set()
    c1.store.save_without_locking = c1_save_without_locking

    def c1_set():
        c1.set('one', 'c1')
    t1 = threading.Thread(target=c1_set)
    self.addCleanup(t1.join)
    self.addCleanup(do_writing.set)
    t1.start()
    ready_to_write.wait()
    self.assertEqual('c1', c1.get('one'))
    c2 = self.get_stack(self)
    self.assertEqual('1', c2.get('one'))
    do_writing.set()
    writing_done.wait()
    c3 = self.get_stack(self)
    self.assertEqual('c1', c3.get('one'))