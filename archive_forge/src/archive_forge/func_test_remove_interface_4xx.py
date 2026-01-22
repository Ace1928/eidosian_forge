from unittest import mock
import testtools
from openstack import exceptions
from openstack.network.v2 import router
from openstack.tests.unit import base
def test_remove_interface_4xx(self):
    sot = router.Router(**EXAMPLE)
    response = mock.Mock()
    msg = '.*borked'
    response.body = {'NeutronError': {'message': msg}}
    response.json = mock.Mock(return_value=response.body)
    response.ok = False
    response.status_code = 409
    response.headers = {'content-type': 'application/json'}
    sess = mock.Mock()
    sess.put = mock.Mock(return_value=response)
    body = {'subnet_id': '3'}
    with testtools.ExpectedException(exceptions.ConflictException, msg):
        sot.remove_interface(sess, **body)