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
def test_get_project_with_parents_as_list_with_invalid_id(self):
    """Call ``GET /projects/{project_id}?parents_as_list``."""
    self.get('/projects/%(project_id)s?parents_as_list' % {'project_id': None}, expected_status=http.client.NOT_FOUND)
    self.get('/projects/%(project_id)s?parents_as_list' % {'project_id': uuid.uuid4().hex}, expected_status=http.client.NOT_FOUND)