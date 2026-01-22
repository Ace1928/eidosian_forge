import uuid
from unittest import mock
from keystone.assignment.core import Manager as AssignmentApi
from keystone.auth.plugins import mapped
from keystone.exception import ProjectNotFound
from keystone.resource.core import Manager as ResourceApi
from keystone.tests import unit
def test_configure_project_domain_with_domain_id(self):
    self.shadow_project_mock['domain'] = self.domain_mock
    mapped.configure_project_domain(self.shadow_project_mock, self.idp_domain_uuid_mock, self.resource_api_mock)
    self.assertIn('domain', self.shadow_project_mock)
    self.assertEqual(self.domain_uuid_mock, self.shadow_project_mock['domain']['id'])