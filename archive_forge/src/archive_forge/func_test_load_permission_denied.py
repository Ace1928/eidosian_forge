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
def test_load_permission_denied(self):
    """Ensure we get warned when trying to load an inaccessible file."""
    warnings = []

    def warning(*args):
        warnings.append(args[0] % args[1:])
    self.overrideAttr(trace, 'warning', warning)
    t = self.get_transport()

    def get_bytes(relpath):
        raise errors.PermissionDenied(relpath, '')
    t.get_bytes = get_bytes
    store = config.TransportIniFileStore(t, 'foo.conf')
    self.assertRaises(errors.PermissionDenied, store.load)
    self.assertEqual(warnings, ['Permission denied while trying to load configuration store %s.' % store.external_url()])