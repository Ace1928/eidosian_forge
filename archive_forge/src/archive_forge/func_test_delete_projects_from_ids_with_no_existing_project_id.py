import copy
from unittest import mock
import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
import keystone.conf
from keystone import exception
from keystone.resource.backends import sql as resource_sql
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import utils as test_utils
def test_delete_projects_from_ids_with_no_existing_project_id(self):
    """Test delete_projects_from_ids issues warning if not found.

        Tests the resource backend call delete_projects_from_ids passing a
        non existing ID in project_ids, which is logged and ignored by
        the backend.
        """
    project_ref = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
    PROVIDERS.resource_api.create_project(project_ref['id'], project_ref)
    projects_ids = (project_ref['id'], uuid.uuid4().hex)
    with mock.patch('keystone.resource.backends.sql.LOG') as mock_log:
        PROVIDERS.resource_api.delete_projects_from_ids(projects_ids)
        self.assertTrue(mock_log.warning.called)
    self.assertRaises(exception.ProjectNotFound, PROVIDERS.resource_api.driver.get_project, project_ref['id'])
    PROVIDERS.resource_api.driver.delete_projects_from_ids([uuid.uuid4().hex])