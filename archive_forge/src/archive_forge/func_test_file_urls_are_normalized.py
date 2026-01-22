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
def test_file_urls_are_normalized(self):
    store = self.get_store(self)
    if sys.platform == 'win32':
        expected_url = 'file:///C:/dir/subdir'
        expected_location = 'C:/dir/subdir'
    else:
        expected_url = 'file:///dir/subdir'
        expected_location = '/dir/subdir'
    matcher = config.LocationMatcher(store, expected_url)
    self.assertEqual(expected_location, matcher.location)