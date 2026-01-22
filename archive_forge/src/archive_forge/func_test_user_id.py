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
def test_user_id(self):
    branch = FakeBranch()
    my_config = config.BranchConfig(branch)
    self.assertIsNot(None, my_config.username())
    my_config.branch.control_files.files['email'] = 'John'
    my_config.set_user_option('email', 'Robert Collins <robertc@example.org>')
    self.assertEqual('Robert Collins <robertc@example.org>', my_config.username())