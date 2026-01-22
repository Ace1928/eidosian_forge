from unittest import mock
import uuid
import testtools
from openstack.config import cloud_region
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
@mock.patch.object(cloud_region.CloudRegion, 'get_session')
def test_get_session_endpoint_identity(self, get_session_mock):
    session_mock = mock.Mock()
    get_session_mock.return_value = session_mock
    self.cloud.get_session_endpoint('identity')
    kwargs = dict(interface='public', region_name='RegionOne', service_name=None, service_type='identity')
    session_mock.get_endpoint.assert_called_with(**kwargs)