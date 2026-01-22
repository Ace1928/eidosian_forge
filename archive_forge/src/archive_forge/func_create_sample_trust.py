import datetime
import uuid
from oslo_utils import timeutils
from keystone.common import provider_api
from keystone import exception
from keystone.tests.unit import core
def create_sample_trust(self, new_id, remaining_uses=None):
    self.trustor = self.user_foo
    self.trustee = self.user_two
    expires_at = datetime.datetime.utcnow().replace(year=2032)
    trust_data = PROVIDERS.trust_api.create_trust(new_id, {'trustor_user_id': self.trustor['id'], 'trustee_user_id': self.user_two['id'], 'project_id': self.project_bar['id'], 'expires_at': expires_at, 'impersonation': True, 'remaining_uses': remaining_uses}, roles=[{'id': 'member'}, {'id': 'other'}, {'id': 'browser'}])
    return trust_data