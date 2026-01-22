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
def test_list_head_domains(self):
    """Call ``GET & HEAD /domains``."""
    resource_url = '/domains'
    r = self.get(resource_url)
    self.assertValidDomainListResponse(r, ref=self.domain, resource_url=resource_url)
    self.head(resource_url, expected_status=http.client.OK)