from unittest import mock
from urllib import parse
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import resource
from heat.engine.resources.openstack.keystone import region
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_region_handle_update(self):
    self.test_region.resource_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    prop_diff = {region.KeystoneRegion.DESCRIPTION: 'Test Region updated', region.KeystoneRegion.ENABLED: False, region.KeystoneRegion.PARENT_REGION: 'test_parent_region'}
    self.test_region.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
    self.regions.update.assert_called_once_with(region=self.test_region.resource_id, description=prop_diff[region.KeystoneRegion.DESCRIPTION], enabled=prop_diff[region.KeystoneRegion.ENABLED], parent_region='test_parent_region')