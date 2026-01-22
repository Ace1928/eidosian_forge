from unittest import mock
import testtools
from openstack import exceptions
from openstack.network.v2 import router
from openstack.tests.unit import base
def test_add_interface_port(self):
    sot = router.Router(**EXAMPLE)
    response = mock.Mock()
    response.body = {'subnet_id': '3', 'port_id': '3'}
    response.json = mock.Mock(return_value=response.body)
    response.status_code = 200
    sess = mock.Mock()
    sess.put = mock.Mock(return_value=response)
    body = {'port_id': '3'}
    self.assertEqual(response.body, sot.add_interface(sess, **body))
    url = 'routers/IDENTIFIER/add_router_interface'
    sess.put.assert_called_with(url, json=body)