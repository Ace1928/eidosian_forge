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
def test_create_store_in_created_dir(self):
    self.assertPathDoesNotExist('dir')
    t = self.get_transport('dir/subdir')
    store = config.LockableIniFileStore(t, 'foo.conf')
    store.get_mutable_section(None).set('foo', 'bar')
    store.save()
    self.assertPathExists('dir/subdir')