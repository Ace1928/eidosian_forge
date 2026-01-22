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
def test_valid_names(self):
    self.assertTrue(self.is_valid('foo'))
    self.assertTrue(self.is_valid('foo.bar'))
    self.assertTrue(self.is_valid('f1'))
    self.assertTrue(self.is_valid('_'))
    self.assertTrue(self.is_valid('__bar__'))
    self.assertTrue(self.is_valid('a_'))
    self.assertTrue(self.is_valid('a1'))
    self.assertTrue(self.is_valid('guessed-layout'))