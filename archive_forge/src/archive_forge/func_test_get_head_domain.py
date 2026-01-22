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
def test_get_head_domain(self):
    """Call ``GET /domains/{domain_id}``."""
    resource_url = '/domains/%(domain_id)s' % {'domain_id': self.domain_id}
    r = self.get(resource_url)
    self.assertValidDomainResponse(r, self.domain)
    self.head(resource_url, expected_status=http.client.OK)