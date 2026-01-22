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
def test_update_project_is_domain_not_allowed(self):
    """Call ``PATCH /projects/{project_id}`` with is_domain.

        The is_domain flag is immutable.
        """
    project = unit.new_project_ref(domain_id=self.domain['id'])
    resp = self.post('/projects', body={'project': project})
    self.assertFalse(resp.result['project']['is_domain'])
    project['parent_id'] = resp.result['project']['parent_id']
    project['is_domain'] = True
    self.patch('/projects/%(project_id)s' % {'project_id': resp.result['project']['id']}, body={'project': project}, expected_status=http.client.BAD_REQUEST)