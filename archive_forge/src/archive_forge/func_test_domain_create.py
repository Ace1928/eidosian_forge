from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from openstackclient.tests.functional.identity.v3 import common
def test_domain_create(self):
    domain_name = data_utils.rand_name('TestDomain')
    raw_output = self.openstack('domain create %s' % domain_name)
    self.addCleanup(self.openstack, 'domain delete %s' % domain_name)
    self.addCleanup(self.openstack, 'domain set --disable %s' % domain_name)
    items = self.parse_show(raw_output)
    self.assert_show_fields(items, self.DOMAIN_FIELDS)