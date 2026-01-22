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
def test_create_domain_case_sensitivity(self):
    """Call `POST /domains`` twice with upper() and lower() cased name."""
    ref = unit.new_domain_ref()
    ref['name'] = ref['name'].lower()
    r = self.post('/domains', body={'domain': ref})
    self.assertValidDomainResponse(r, ref)
    ref['name'] = ref['name'].upper()
    r = self.post('/domains', body={'domain': ref})
    self.assertValidDomainResponse(r, ref)