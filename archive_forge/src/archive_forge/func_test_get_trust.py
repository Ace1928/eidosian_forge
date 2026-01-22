import datetime
import uuid
from oslo_utils import timeutils
from keystone.common import provider_api
from keystone import exception
from keystone.tests.unit import core
def test_get_trust(self):
    new_id = uuid.uuid4().hex
    trust_data = self.create_sample_trust(new_id)
    trust_id = trust_data['id']
    self.assertIsNotNone(trust_data)
    trust_data = PROVIDERS.trust_api.get_trust(trust_id)
    self.assertEqual(new_id, trust_data['id'])
    PROVIDERS.trust_api.delete_trust(trust_data['id'])