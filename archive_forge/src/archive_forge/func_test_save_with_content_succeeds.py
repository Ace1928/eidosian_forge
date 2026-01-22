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
def test_save_with_content_succeeds(self):
    if self.store_id in ('branch', 'remote_branch'):
        raise tests.TestNotApplicable('branch.conf is *always* created when a branch is initialized')
    store = self.get_store(self)
    store._load_from_string(b'foo=bar\n')
    self.assertEqual(False, self.has_store(store))
    store.save()
    self.assertEqual(True, self.has_store(store))
    modified_store = self.get_store(self)
    sections = list(modified_store.get_sections())
    self.assertLength(1, sections)
    self.assertSectionContent((None, {'foo': 'bar'}), sections[0])