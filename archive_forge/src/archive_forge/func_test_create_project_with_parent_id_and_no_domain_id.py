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
@test_utils.wip('waiting for support for parent_id to imply domain_id')
def test_create_project_with_parent_id_and_no_domain_id(self):
    """Call ``POST /projects``."""
    ref_child = unit.new_project_ref(parent_id=self.project['id'])
    r = self.post('/projects', body={'project': ref_child})
    self.assertEqual(self.project['domain_id'], r.result['project']['domain_id'])
    ref_child['domain_id'] = self.domain['id']
    self.assertValidProjectResponse(r, ref_child)