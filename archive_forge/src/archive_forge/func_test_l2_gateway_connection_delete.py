from unittest import mock
from neutronclient.v2_0 import client as neutronclient
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_l2_gateway_connection_delete(self):
    self._create_l2_gateway_connection()
    self.stack.delete()
    self.mockclient.delete_l2_gateway_connection.assert_called_with('e491171c-3458-4d85-b3a3-68a7c4a1cacd')