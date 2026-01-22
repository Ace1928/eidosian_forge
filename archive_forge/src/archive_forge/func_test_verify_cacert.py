import copy
from unittest import mock
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session as ksa_session
from openstack.config import cloud_region
from openstack.config import defaults
from openstack import exceptions
from openstack.tests.unit.config import base
from openstack import version as openstack_version
def test_verify_cacert(self):
    config_dict = copy.deepcopy(fake_config_dict)
    config_dict['cacert'] = 'certfile'
    config_dict['verify'] = False
    cc = cloud_region.CloudRegion('test1', 'region-xx', config_dict)
    verify, cert = cc.get_requests_verify_args()
    self.assertFalse(verify)
    config_dict['verify'] = True
    cc = cloud_region.CloudRegion('test1', 'region-xx', config_dict)
    verify, cert = cc.get_requests_verify_args()
    self.assertEqual('certfile', verify)
    config_dict['insecure'] = True
    cc = cloud_region.CloudRegion('test1', 'region-xx', config_dict)
    verify, cert = cc.get_requests_verify_args()
    self.assertEqual(False, verify)