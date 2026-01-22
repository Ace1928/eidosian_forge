import uuid
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
def test_policy_duplicate_conflict_gives_name(self):
    policy_ref = unit.new_policy_ref()
    PROVIDERS.policy_api.create_policy(policy_ref['id'], policy_ref)
    try:
        PROVIDERS.policy_api.create_policy(policy_ref['id'], policy_ref)
    except exception.Conflict as e:
        self.assertIn('Duplicate entry found with name %s' % policy_ref['name'], repr(e))
    else:
        self.fail('Create duplicate policy did not raise a conflict')