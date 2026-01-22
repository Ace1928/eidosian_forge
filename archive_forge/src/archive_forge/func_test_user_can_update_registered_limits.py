import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_update_registered_limits(self):
    service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
    registered_limit = unit.new_registered_limit_ref(service_id=service['id'], id=uuid.uuid4().hex)
    limits = PROVIDERS.unified_limit_api.create_registered_limits([registered_limit])
    limit_id = limits[0]['id']
    with self.test_client() as c:
        update = {'registered_limit': {'default_limit': 5}}
        c.patch('/v3/registered_limits/%s' % limit_id, json=update, headers=self.headers)