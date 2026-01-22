import copy
from unittest import mock
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session as ksa_session
from openstack.config import cloud_region
from openstack.config import defaults
from openstack import exceptions
from openstack.tests.unit.config import base
from openstack import version as openstack_version
def test_get_session_no_auth(self):
    config_dict = defaults.get_defaults()
    config_dict.update(fake_services_dict)
    cc = cloud_region.CloudRegion('test1', 'region-al', config_dict)
    self.assertRaises(exceptions.ConfigException, cc.get_session)