from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional.identity.v3 import common
def test_user_set_default_project_id(self):
    username = self._create_dummy_user()
    project_name = self._create_dummy_project()
    raw_output = self.openstack('user show --domain %(domain)s %(name)s' % {'domain': self.domain_name, 'name': username})
    user = self.parse_show_as_object(raw_output)
    raw_output = self.openstack('user set --project %(project)s --project-domain %(project_domain)s %(id)s' % {'project': project_name, 'project_domain': self.domain_name, 'id': user['id']})
    self.assertEqual(0, len(raw_output))
    raw_output = self.openstack('user show --domain %(domain)s %(name)s' % {'domain': self.domain_name, 'name': username})
    updated_user = self.parse_show_as_object(raw_output)
    raw_output = self.openstack('project show --domain %(domain)s %(name)s' % {'domain': self.domain_name, 'name': project_name})
    project = self.parse_show_as_object(raw_output)
    self.assertEqual(user['id'], updated_user['id'])
    self.assertEqual(project['id'], updated_user['default_project_id'])