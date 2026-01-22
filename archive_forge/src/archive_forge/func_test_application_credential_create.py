import datetime
from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional.identity.v3 import common
def test_application_credential_create(self):
    name = data_utils.rand_name('name')
    raw_output = self.openstack('application credential create %(name)s' % {'name': name})
    self.addCleanup(self.openstack, 'application credential delete %(name)s' % {'name': name})
    items = self.parse_show(raw_output)
    self.assert_show_fields(items, self.APPLICATION_CREDENTIAL_FIELDS)