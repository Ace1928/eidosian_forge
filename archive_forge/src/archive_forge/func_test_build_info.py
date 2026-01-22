from oslo_utils import reflection
import heat.api.openstack.v1 as api_v1
from heat.tests import common
def test_build_info(self):
    self.assertRoute(self.m, '/fake_tenant/build_info', 'GET', 'build_info', 'BuildInfoController', {'tenant_id': 'fake_tenant'})