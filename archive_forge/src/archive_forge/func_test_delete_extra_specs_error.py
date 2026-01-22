from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v3 import type
from openstack import exceptions
from openstack.tests.unit import base
def test_delete_extra_specs_error(self):
    sess = mock.Mock()
    response = mock.Mock()
    response.status_code = 400
    response.content = None
    sess.delete.return_value = response
    sot = type.Type(id=FAKE_ID)
    key = 'hey'
    self.assertRaises(exceptions.BadRequestException, sot.delete_extra_specs, sess, [key])