from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import flavor_access
def test_repr_flavor_access(self):
    flavor = self.cs.flavors.get(2)
    tenant = 'proj3'
    r = self.cs.flavor_access.add_tenant_access(flavor, tenant)

    def get_expected(flavor_access):
        return '<FlavorAccess flavor id: %s, tenant id: %s>' % (flavor_access.flavor_id, flavor_access.tenant_id)
    for a in r:
        self.assertEqual(get_expected(a), repr(a))