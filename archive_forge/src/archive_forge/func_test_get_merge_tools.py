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
def test_get_merge_tools(self):
    conf = self._get_sample_config()
    tools = conf.get_merge_tools()
    self.log(repr(tools))
    self.assertEqual({'funkytool': 'funkytool "arg with spaces" {this_temp}', 'sometool': 'sometool {base} {this} {other} -o {result}', 'newtool': '"newtool with spaces" {this_temp}'}, tools)