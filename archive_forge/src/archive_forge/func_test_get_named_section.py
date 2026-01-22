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
def test_get_named_section(self):
    store = self.get_store(self)
    store._load_from_string(b'[baz]\nfoo=bar')
    sections = list(store.get_sections())
    self.assertLength(1, sections)
    self.assertSectionContent(('baz', {'foo': 'bar'}), sections[0])