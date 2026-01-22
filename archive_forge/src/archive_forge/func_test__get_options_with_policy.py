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
def test__get_options_with_policy(self):
    self.get_branch_config('/dir/subdir', location_config='[/dir]\nother_url = /other-dir\nother_url:policy = appendpath\n[/dir/subdir]\nother_url = /other-subdir\n')
    self.assertOptions([('other_url', '/other-subdir', '/dir/subdir', 'locations'), ('other_url', '/other-dir', '/dir', 'locations'), ('other_url:policy', 'appendpath', '/dir', 'locations')], self.my_location_config)