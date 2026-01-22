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
def test_update_domain_updates_is_domain_project(self):
    """Check the project that acts as a domain is updated.

        Call ``PATCH /domains``.
        """
    domain_ref = unit.new_domain_ref()
    r = self.post('/domains', body={'domain': domain_ref})
    self.assertValidDomainResponse(r, domain_ref)
    self.patch('/domains/%s' % r.result['domain']['id'], body={'domain': {'enabled': False}})
    r = self.get('/projects/%(project_id)s' % {'project_id': r.result['domain']['id']})
    self.assertValidProjectResponse(r)
    self.assertFalse(r.result['project']['enabled'])