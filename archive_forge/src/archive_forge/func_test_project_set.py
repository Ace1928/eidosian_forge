from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional.identity.v2 import common
def test_project_set(self):
    project_name = self._create_dummy_project()
    new_project_name = data_utils.rand_name('NewTestProject')
    raw_output = self.openstack('project set --name %(new_name)s --disable --property k0=v0 %(name)s' % {'new_name': new_project_name, 'name': project_name})
    self.assertEqual(0, len(raw_output))
    raw_output = self.openstack('project show %s' % new_project_name)
    items = self.parse_show(raw_output)
    fields = list(self.PROJECT_FIELDS)
    fields.extend(['properties'])
    self.assert_show_fields(items, fields)
    project = self.parse_show_as_object(raw_output)
    self.assertEqual(new_project_name, project['name'])
    self.assertFalse(project['enabled'])
    self.assertEqual("k0='v0'", project['properties'])