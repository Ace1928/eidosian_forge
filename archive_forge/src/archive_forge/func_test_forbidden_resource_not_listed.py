from heatclient import exc
import keystoneclient
from heat_integrationtests.functional import functional_base
def test_forbidden_resource_not_listed(self):
    resources = self.client.resource_types.list()
    self.assertNotIn(self.forbidden_r_type, (r.resource_type for r in resources))