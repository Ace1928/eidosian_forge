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
def test_expand_relpath_unknonw_in_global(self):
    g_store = config.GlobalStore()
    g_store._load_from_string(b'\n[DEFAULT]\ngfoo = {relpath}\n')
    g_store.save()
    stack = config.LocationStack('/home/user/project/branch')
    self.assertRaises(config.ExpandingUnknownOption, stack.get, 'gfoo', expand=True)