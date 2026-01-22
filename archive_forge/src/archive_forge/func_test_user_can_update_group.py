import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import group as gp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_user_can_update_group(self):
    domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
    group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=domain['id']))
    update = {'group': {'description': uuid.uuid4().hex}}
    with self.test_client() as c:
        c.patch('/v3/groups/%s' % group['id'], json=update, headers=self.headers)