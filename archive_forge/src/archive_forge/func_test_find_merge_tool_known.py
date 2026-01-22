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
def test_find_merge_tool_known(self):
    conf = self._get_empty_config()
    cmdline = conf.find_merge_tool('kdiff3')
    self.assertEqual('kdiff3 {base} {this} {other} -o {result}', cmdline)