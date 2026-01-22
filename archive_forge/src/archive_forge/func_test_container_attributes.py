from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.barbican import order
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_container_attributes(self):
    mock_order = mock.Mock()
    mock_order.container_ref = 'test-container-ref'
    mock_container = mock.Mock()
    mock_container.public_key = mock.Mock(payload='public-key')
    mock_container.private_key = mock.Mock(payload='private-key')
    mock_container.certificate = mock.Mock(payload='cert')
    mock_container.intermediates = mock.Mock(payload='interm')
    res = self._create_resource('foo', self.res_template, self.stack)
    self.barbican.orders.get.return_value = mock_order
    self.barbican.containers.get.return_value = mock_container
    self.assertEqual('public-key', res.FnGetAtt('public_key'))
    self.barbican.containers.get.assert_called_once_with('test-container-ref')
    self.assertEqual('private-key', res.FnGetAtt('private_key'))
    self.assertEqual('cert', res.FnGetAtt('certificate'))
    self.assertEqual('interm', res.FnGetAtt('intermediates'))