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
def test_update_project_with_tags(self):
    project, tags = self._create_project_and_tags(num_of_tags=9)
    tag = uuid.uuid4().hex
    project['tags'].append(tag)
    ref = self.patch('/projects/%(project_id)s' % {'project_id': self.project_id}, body={'project': {'tags': project['tags']}})
    self.assertIn(tag, ref.result['project']['tags'])