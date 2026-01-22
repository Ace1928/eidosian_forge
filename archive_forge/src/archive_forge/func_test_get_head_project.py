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
def test_get_head_project(self):
    """Call ``GET & HEAD /projects/{project_id}``."""
    resource_url = '/projects/%(project_id)s' % {'project_id': self.project_id}
    r = self.get(resource_url)
    self.assertValidProjectResponse(r, self.project)
    self.head(resource_url, expected_status=http.client.OK)