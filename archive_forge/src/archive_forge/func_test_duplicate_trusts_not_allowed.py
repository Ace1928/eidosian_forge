import datetime
import uuid
from oslo_utils import timeutils
from keystone.common import provider_api
from keystone import exception
from keystone.tests.unit import core
def test_duplicate_trusts_not_allowed(self):
    self.trustor = self.user_foo
    self.trustee = self.user_two
    trust_data = {'trustor_user_id': self.trustor['id'], 'trustee_user_id': self.user_two['id'], 'project_id': self.project_bar['id'], 'expires_at': timeutils.parse_isotime('2032-02-18T18:10:00Z'), 'impersonation': True, 'remaining_uses': None}
    roles = [{'id': 'member'}, {'id': 'other'}, {'id': 'browser'}]
    PROVIDERS.trust_api.create_trust(uuid.uuid4().hex, trust_data, roles)
    self.assertRaises(exception.Conflict, PROVIDERS.trust_api.create_trust, uuid.uuid4().hex, trust_data, roles)