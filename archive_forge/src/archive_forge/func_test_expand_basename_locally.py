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
def test_expand_basename_locally(self):
    l_store = config.LocationStore()
    l_store._load_from_string(b'\n[/home/user/project]\nbfoo = {basename}\n')
    l_store.save()
    stack = config.LocationStack('/home/user/project/branch')
    self.assertEqual('branch', stack.get('bfoo', expand=True))