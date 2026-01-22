from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.barbican import order
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_attributes_handle_exceptions(self):
    mock_order = mock.Mock()
    res = self._create_resource('foo', self.res_template, self.stack)
    self.barbican.orders.get.return_value = mock_order
    self.barbican.barbican_client.HTTPClientError = Exception
    self.barbican.orders.get.side_effect = Exception('boom')
    self.assertRaises(self.barbican.barbican_client.HTTPClientError, res.FnGetAtt, 'order_ref')