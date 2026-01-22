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
def test_update_project_parent_id(self):
    """Call ``PATCH /projects/{project_id}``."""
    projects = self._create_projects_hierarchy()
    leaf_project = projects[1]['project']
    leaf_project['parent_id'] = None
    self.patch('/projects/%(project_id)s' % {'project_id': leaf_project['id']}, body={'project': leaf_project}, expected_status=http.client.FORBIDDEN)