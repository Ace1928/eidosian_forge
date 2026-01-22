import uuid
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
def test_user_duplicate_conflict_gives_name(self):
    user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    user = PROVIDERS.identity_api.create_user(user)
    user['id'] = uuid.uuid4().hex
    try:
        PROVIDERS.identity_api.create_user(user)
    except exception.Conflict as e:
        self.assertIn('Duplicate entry found with name %s' % user['name'], repr(e))
    else:
        self.fail('Create duplicate user did not raise a conflict')