from unittest import mock
import testtools
from openstack import exceptions
from openstack.network.v2 import router
from openstack.tests.unit import base
def test_remove_interface_port(self):
    sot = router.Router(**EXAMPLE)
    response = mock.Mock()
    response.body = {'subnet_id': '3', 'port_id': '3'}
    response.json = mock.Mock(return_value=response.body)
    response.status_code = 200
    sess = mock.Mock()
    sess.put = mock.Mock(return_value=response)
    body = {'network_id': 3, 'enable_snat': True}
    self.assertEqual(response.body, sot.remove_interface(sess, **body))
    url = 'routers/IDENTIFIER/remove_router_interface'
    sess.put.assert_called_with(url, json=body)