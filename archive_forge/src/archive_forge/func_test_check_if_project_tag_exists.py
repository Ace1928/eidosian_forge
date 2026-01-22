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
def test_check_if_project_tag_exists(self):
    project, tags = self._create_project_and_tags(num_of_tags=5)
    self.head('/projects/%(project_id)s/tags/%(value)s' % {'project_id': project['id'], 'value': tags[0]}, expected_status=http.client.NO_CONTENT)