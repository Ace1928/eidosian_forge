import copy
from unittest import mock
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session as ksa_session
from openstack.config import cloud_region
from openstack.config import defaults
from openstack import exceptions
from openstack.tests.unit.config import base
from openstack import version as openstack_version
def test_cert_with_key(self):
    config_dict = copy.deepcopy(fake_config_dict)
    config_dict['cacert'] = None
    config_dict['verify'] = False
    config_dict['cert'] = 'cert'
    config_dict['key'] = 'key'
    cc = cloud_region.CloudRegion('test1', 'region-xx', config_dict)
    verify, cert = cc.get_requests_verify_args()
    self.assertEqual(('cert', 'key'), cert)