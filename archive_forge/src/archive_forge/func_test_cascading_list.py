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
def test_cascading_list(self):
    self.conf.store._load_from_string(b'\nfoo=start,{bar}\nbar=middle,{baz}\nbaz=end\nlist={foo}\n')
    self.registry.register(config.ListOption('list'))
    self.registry.register(config.ListOption('baz'))
    self.assertEqual(['start', 'middle', 'end'], self.conf.get('list', expand=True))