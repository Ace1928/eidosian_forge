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
def test_simple_store(self):
    store = config.IniFileStore()
    store._load_from_string(b'foo=bar')
    conf = config.Stack([store.get_sections])
    tuples = list(conf.iter_sections())
    self.assertLength(1, tuples)
    found_store, found_section = tuples[0]
    self.assertIs(store, found_store)