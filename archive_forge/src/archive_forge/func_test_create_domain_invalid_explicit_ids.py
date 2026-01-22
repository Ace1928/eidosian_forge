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
def test_create_domain_invalid_explicit_ids(self):
    """Call ``POST /domains`` with various invalid explicit_domain_ids."""
    ref = unit.new_domain_ref()
    bad_ids = ['bad!', '', '9aea63518f0040c', '1234567890123456789012345678901234567890', '9aea63518f0040c6b4518d8d2242911c9aea63518f0040c6b45']
    for explicit_domain_id in bad_ids:
        ref['explicit_domain_id'] = explicit_domain_id
        self.post('/domains', body={'domain': {}}, expected_status=http.client.BAD_REQUEST)