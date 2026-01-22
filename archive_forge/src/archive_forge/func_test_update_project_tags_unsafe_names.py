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
def test_update_project_tags_unsafe_names(self):
    project, tags = self._create_project_and_tags(num_of_tags=5)
    invalid_chars = [',', '/']
    for char in invalid_chars:
        tags[0] = uuid.uuid4().hex + char
        self.put('/projects/%(project_id)s/tags' % {'project_id': project['id']}, body={'tags': tags}, expected_status=http.client.BAD_REQUEST)