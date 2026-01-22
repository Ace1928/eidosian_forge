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
def test_cannot_list_grants_for_group_own_domain_on_other_domain(self):
    group_domain_id = self.domain_id
    domain_id = CONF.identity.default_domain_id
    group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=group_domain_id))
    PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, group_id=group['id'], domain_id=domain_id)
    with self.test_client() as c:
        c.get('/v3/domains/%s/groups/%s/roles' % (domain_id, group['id']), headers=self.headers, expected_status_code=http.client.FORBIDDEN)