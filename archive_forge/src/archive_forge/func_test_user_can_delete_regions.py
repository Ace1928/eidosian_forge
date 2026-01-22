import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_delete_regions(self):
    region = PROVIDERS.catalog_api.create_region(unit.new_region_ref())
    with self.test_client() as c:
        c.delete('/v3/regions/%s' % region['id'], headers=self.headers)