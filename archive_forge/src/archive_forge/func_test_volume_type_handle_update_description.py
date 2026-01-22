import collections
from unittest import mock
from heat.common import exception
from heat.engine.clients.os import cinder as c_plugin
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_volume_type_handle_update_description(self):
    update_args = {'description': 'update'}
    self._test_update(update_args)