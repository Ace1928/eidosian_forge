import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
from keystone.tests.unit import utils as test_utils
def test_get_project_with_parents_as_list_and_parents_as_ids(self):
    """Attempt to list a project's parents as both a list and as IDs.

        This uses ``GET /projects/{project_id}?parents_as_list&parents_as_ids``
        which should fail with a Bad Request due to the conflicting query
        strings.

        """
    projects = self._create_projects_hierarchy(hierarchy_size=2)
    self.get('/projects/%(project_id)s?parents_as_list&parents_as_ids' % {'project_id': projects[1]['project']['id']}, expected_status=http.client.BAD_REQUEST)