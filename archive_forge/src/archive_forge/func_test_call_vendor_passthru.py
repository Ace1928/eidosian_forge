from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import driver
from openstack import exceptions
from openstack.tests.unit import base
@mock.patch.object(exceptions, 'raise_from_response', mock.Mock())
def test_call_vendor_passthru(self):
    self.session = mock.Mock(spec=adapter.Adapter)
    sot = driver.Driver(**FAKE)
    sot.call_vendor_passthru(self.session, 'GET', 'fake_vendor_method')
    self.session.get.assert_called_once_with('drivers/{}/vendor_passthru?method={}'.format(FAKE['name'], 'fake_vendor_method'), json=None, headers=mock.ANY, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)
    sot.call_vendor_passthru(self.session, 'PUT', 'fake_vendor_method', body={'fake_param_key': 'fake_param_value'})
    self.session.put.assert_called_once_with('drivers/{}/vendor_passthru?method={}'.format(FAKE['name'], 'fake_vendor_method'), json={'fake_param_key': 'fake_param_value'}, headers=mock.ANY, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)
    sot.call_vendor_passthru(self.session, 'POST', 'fake_vendor_method', body={'fake_param_key': 'fake_param_value'})
    self.session.post.assert_called_once_with('drivers/{}/vendor_passthru?method={}'.format(FAKE['name'], 'fake_vendor_method'), json={'fake_param_key': 'fake_param_value'}, headers=mock.ANY, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)
    sot.call_vendor_passthru(self.session, 'DELETE', 'fake_vendor_method')
    self.session.delete.assert_called_once_with('drivers/{}/vendor_passthru?method={}'.format(FAKE['name'], 'fake_vendor_method'), json=None, headers=mock.ANY, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)