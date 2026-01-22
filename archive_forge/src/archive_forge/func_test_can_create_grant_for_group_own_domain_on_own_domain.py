import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import grant as gp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_can_create_grant_for_group_own_domain_on_own_domain(self):
    group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=self.domain_id))
    with self.test_client() as c:
        c.put('/v3/domains/%s/groups/%s/roles/%s' % (self.domain_id, group['id'], self.bootstrapper.reader_role_id), headers=self.headers)