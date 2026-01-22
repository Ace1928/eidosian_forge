import datetime
from unittest import mock
import uuid
from keystone.common.cache import _context_cache
from keystone.common import utils as ks_utils
from keystone import exception
from keystone.models import token_model
from keystone.tests.unit import base_classes
def test_serialize_and_deserialize_token_model(self):
    serialized = self.token_handler.serialize(self.exp_token)
    token = self.token_handler.deserialize(serialized)
    self.assertEqual(self.exp_token.user_id, token.user_id)
    self.assertEqual(self.exp_token.project_id, token.project_id)
    self.assertEqual(self.exp_token.id, token.id)
    self.assertEqual(self.exp_token.issued_at, token.issued_at)