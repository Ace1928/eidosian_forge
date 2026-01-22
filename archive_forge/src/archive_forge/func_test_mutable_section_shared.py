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
def test_mutable_section_shared(self):
    store = self.get_store(self)
    store._load_from_string(b'foo=bar\n')
    if self.store_id in ('branch', 'remote_branch'):
        self.addCleanup(store.branch.lock_write().unlock)
    section1 = store.get_mutable_section(None)
    section2 = store.get_mutable_section(None)
    self.assertIs(section1, section2)