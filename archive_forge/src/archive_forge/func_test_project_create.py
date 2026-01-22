from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional.identity.v2 import common
def test_project_create(self):
    project_name = data_utils.rand_name('TestProject')
    description = data_utils.rand_name('description')
    raw_output = self.openstack('project create --description %(description)s --enable --property k1=v1 --property k2=v2 %(name)s' % {'description': description, 'name': project_name})
    self.addCleanup(self.openstack, 'project delete %s' % project_name)
    items = self.parse_show(raw_output)
    show_fields = list(self.PROJECT_FIELDS)
    show_fields.extend(['k1', 'k2'])
    self.assert_show_fields(items, show_fields)
    project = self.parse_show_as_object(raw_output)
    self.assertEqual('v1', project['k1'])
    self.assertEqual('v2', project['k2'])