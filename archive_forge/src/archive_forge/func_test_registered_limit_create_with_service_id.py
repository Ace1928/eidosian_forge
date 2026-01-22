import os
from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional.identity.v3 import common
def test_registered_limit_create_with_service_id(self):
    service_name = self._create_dummy_service()
    raw_output = self.openstack('service show %(service_name)s' % {'service_name': service_name})
    service_items = self.parse_show(raw_output)
    service_id = self._extract_value_from_items('id', service_items)
    raw_output = self.openstack('registered limit create --service %(service_id)s --default-limit %(default_limit)s %(resource_name)s' % {'service_id': service_id, 'default_limit': 10, 'resource_name': 'cores'}, cloud=SYSTEM_CLOUD)
    items = self.parse_show(raw_output)
    registered_limit_id = self._extract_value_from_items('id', items)
    self.addCleanup(self.openstack, 'registered limit delete %(registered_limit_id)s' % {'registered_limit_id': registered_limit_id}, cloud=SYSTEM_CLOUD)
    self.assert_show_fields(items, self.REGISTERED_LIMIT_FIELDS)