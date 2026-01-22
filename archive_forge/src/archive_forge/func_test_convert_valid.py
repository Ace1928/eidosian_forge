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
def test_convert_valid(self):
    registry = _mod_registry.Registry()
    registry.register('someval', 1234)
    opt = self.get_option(registry)
    self.assertConverted(1234, opt, 'someval')
    self.assertConverted(1234, opt, 'someval')
    self.assertConverted(None, opt, None)