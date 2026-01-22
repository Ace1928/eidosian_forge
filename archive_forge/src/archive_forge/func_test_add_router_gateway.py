from unittest import mock
import testtools
from openstack import exceptions
from openstack.network.v2 import router
from openstack.tests.unit import base
def test_add_router_gateway(self):
    sot = router.Router(**EXAMPLE_WITH_OPTIONAL)
    response = mock.Mock()
    response.body = {'network_id': '3', 'enable_snat': True}
    response.json = mock.Mock(return_value=response.body)
    sess = mock.Mock()
    sess.put = mock.Mock(return_value=response)
    body = {'network_id': 3, 'enable_snat': True}
    self.assertEqual(response.body, sot.add_gateway(sess, **body))
    url = 'routers/IDENTIFIER/add_gateway_router'
    sess.put.assert_called_with(url, json=body)