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
def test_list_project_tags_for_project_with_no_tags(self):
    resp = self.get('/projects/%(project_id)s/tags' % {'project_id': self.project_id}, expected_status=http.client.OK)
    self.assertEqual([], resp.result['tags'])