import copy
from unittest import mock
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session as ksa_session
from openstack.config import cloud_region
from openstack.config import defaults
from openstack import exceptions
from openstack.tests.unit.config import base
from openstack import version as openstack_version
@mock.patch.object(ksa_session, 'Session')
def test_override_session_endpoint_override(self, mock_session):
    config_dict = defaults.get_defaults()
    config_dict.update(fake_services_dict)
    cc = cloud_region.CloudRegion('test1', 'region-al', config_dict, auth_plugin=mock.Mock())
    self.assertEqual(cc.get_session_endpoint('compute'), fake_services_dict['compute_endpoint_override'])