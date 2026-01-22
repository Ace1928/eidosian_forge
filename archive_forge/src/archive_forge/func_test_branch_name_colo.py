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
def test_branch_name_colo(self):
    store = self.get_store(self)
    store._load_from_string(dedent('            [/]\n            push_location=my{branchname}\n        ').encode('ascii'))
    matcher = config.LocationMatcher(store, 'file:///,branch=example%3c')
    self.assertEqual('example<', matcher.branch_name)
    (_, section), = matcher.get_sections()
    self.assertEqual('example<', section.locals['branchname'])