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
def test_get_session_with_timing(self, mock_session):
    fake_session = mock.Mock()
    fake_session.additional_user_agent = []
    mock_session.return_value = fake_session
    config_dict = defaults.get_defaults()
    config_dict.update(fake_services_dict)
    config_dict['timing'] = True
    cc = cloud_region.CloudRegion('test1', 'region-al', config_dict, auth_plugin=mock.Mock())
    cc.get_session()
    mock_session.assert_called_with(auth=mock.ANY, verify=True, cert=None, timeout=None, collect_timing=True, discovery_cache=None)
    self.assertEqual(fake_session.additional_user_agent, [('openstacksdk', openstack_version.__version__)])