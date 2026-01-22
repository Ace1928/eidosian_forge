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
def test_user_can_get_a_domain(self):
    with self.test_client() as c:
        r = c.get('/v3/domains/%s' % self.domain_id, headers=self.headers)
        self.assertEqual(self.domain_id, r.json['domain']['id'])