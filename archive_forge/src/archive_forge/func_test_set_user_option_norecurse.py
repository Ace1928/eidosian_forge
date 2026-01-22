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
def test_set_user_option_norecurse(self):
    self.get_branch_config('http://www.example.com')
    self.my_config.set_user_option('foo', 'bar', store=config.STORE_LOCATION_NORECURSE)
    self.assertEqual(self.my_location_config._get_option_policy('http://www.example.com', 'foo'), config.POLICY_NORECURSE)