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
def test_register_lazy(self):
    r = config.CredentialStoreRegistry()
    r.register_lazy('stub', 'breezy.tests.test_config', 'StubCredentialStore', fallback=False)
    self.assertEqual(['stub'], r.keys())
    self.assertIsInstance(r.get_credential_store('stub'), StubCredentialStore)