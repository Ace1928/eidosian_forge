from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional.identity.v3 import common
def test_service_set(self):
    service_name = self._create_dummy_service()
    new_service_name = data_utils.rand_name('NewTestService')
    new_service_description = data_utils.rand_name('description')
    new_service_type = data_utils.rand_name('NewTestType')
    raw_output = self.openstack('service set --type %(type)s --name %(name)s --description %(description)s --disable %(service)s' % {'type': new_service_type, 'name': new_service_name, 'description': new_service_description, 'service': service_name})
    self.assertEqual(0, len(raw_output))
    raw_output = self.openstack('service show %s' % new_service_name)
    service = self.parse_show_as_object(raw_output)
    self.assertEqual(new_service_type, service['type'])
    self.assertEqual(new_service_name, service['name'])
    self.assertEqual(new_service_description, service['description'])