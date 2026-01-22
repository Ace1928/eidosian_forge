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
def test_get_first_definition(self):
    store1 = config.IniFileStore()
    store1._load_from_string(b'foo=bar')
    store2 = config.IniFileStore()
    store2._load_from_string(b'foo=baz')
    conf = config.Stack([store1.get_sections, store2.get_sections])
    self.assertEqual('bar', conf.get('foo'))