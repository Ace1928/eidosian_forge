from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import cinder as c_plugin
from heat.engine.clients.os import keystone as k_plugin
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_invalid_snapshots(self):
    self._test_invalid_property('snapshots')