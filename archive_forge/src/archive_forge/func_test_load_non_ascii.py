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
def test_load_non_ascii(self):
    """Ensure we display a proper error on non-ascii, non utf-8 content."""
    t = self.get_transport()
    t.put_bytes('foo.conf', b'user=foo\n#%s\n' % (self.invalid_utf8_char,))
    store = config.TransportIniFileStore(t, 'foo.conf')
    self.assertRaises(config.ConfigContentError, store.load)