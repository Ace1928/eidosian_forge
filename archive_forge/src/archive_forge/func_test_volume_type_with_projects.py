import collections
from unittest import mock
from heat.common import exception
from heat.engine.clients.os import cinder as c_plugin
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_volume_type_with_projects(self):
    self.cinderclient.volume_api_version = 3
    self._test_handle_create(projects=['id1', 'id2'])