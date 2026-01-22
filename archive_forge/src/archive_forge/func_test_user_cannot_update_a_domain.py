import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import domain as dp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_user_cannot_update_a_domain(self):
    domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
    update = {'domain': {'description': uuid.uuid4().hex}}
    with self.test_client() as c:
        c.patch('/v3/domains/%s' % domain['id'], json=update, headers=self.headers, expected_status_code=http.client.FORBIDDEN)