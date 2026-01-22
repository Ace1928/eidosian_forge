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
def test_fallback_credentials(self):
    r = config.CredentialStoreRegistry()
    store = StubCredentialStore()
    store.add_credentials('http', 'example.com', 'somebody', 'geheim')
    r.register('stub', store, fallback=True)
    creds = r.get_fallback_credentials('http', 'example.com')
    self.assertEqual('somebody', creds['user'])
    self.assertEqual('geheim', creds['password'])