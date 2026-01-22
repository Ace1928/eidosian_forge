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
def test_list_project_is_domain_filter_default(self):
    """Default project list should not see projects acting as domains."""
    r = self.get('/projects?is_domain=False', expected_status=200)
    number_is_domain_false = len(r.result['projects'])
    new_is_domain_project = unit.new_project_ref(is_domain=True)
    new_is_domain_project = PROVIDERS.resource_api.create_project(new_is_domain_project['id'], new_is_domain_project)
    r = self.get('/projects', expected_status=200)
    self.assertThat(r.result['projects'], matchers.HasLength(number_is_domain_false))
    self.assertNotIn(new_is_domain_project, r.result['projects'])