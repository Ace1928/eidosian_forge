import datetime
import uuid
from oslo_utils import timeutils
from keystone.common import provider_api
from keystone import exception
from keystone.tests.unit import core
def test_list_trusts(self):
    for i in range(3):
        self.create_sample_trust(uuid.uuid4().hex)
    trusts = PROVIDERS.trust_api.list_trusts()
    self.assertEqual(3, len(trusts))