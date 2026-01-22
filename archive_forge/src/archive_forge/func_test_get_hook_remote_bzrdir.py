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
def test_get_hook_remote_bzrdir(self):
    remote_bzrdir = controldir.ControlDir.open(self.get_url('tree'))
    conf = remote_bzrdir._get_config()
    conf.set_option('remotedir', 'file')
    self.assertGetHook(conf, 'file', 'remotedir')