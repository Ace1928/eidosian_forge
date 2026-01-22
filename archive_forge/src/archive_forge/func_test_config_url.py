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
def test_config_url(self):
    """The Branch.get_config will use section that uses a local url"""
    branch = self.make_branch('branch')
    self.assertEqual('branch', branch.nick)
    local_url = urlutils.local_path_to_url('branch')
    conf = config.LocationConfig.from_string('[{}]\nnickname = foobar'.format(local_url), local_url, save=True)
    self.assertIsNot(None, conf)
    self.assertEqual('foobar', branch.nick)