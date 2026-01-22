import datetime
import uuid
from oslo_utils import timeutils
from keystone.common import provider_api
from keystone import exception
from keystone.tests.unit import core
def test_trust_has_remaining_uses_positive(self):
    trust_data = self.create_sample_trust(uuid.uuid4().hex, remaining_uses=5)
    self.assertEqual(5, trust_data['remaining_uses'])
    trust_data = self.create_sample_trust(uuid.uuid4().hex)
    self.assertIsNone(trust_data['remaining_uses'])