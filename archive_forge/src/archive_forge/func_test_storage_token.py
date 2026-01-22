import itertools
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from keystonemiddleware.auth_token import _request
from keystonemiddleware.tests.unit import utils
def test_storage_token(self):
    storage_token = uuid.uuid4().hex
    user_token = uuid.uuid4().hex
    self.assertIsNone(self.request.user_token)
    self.request.headers['X-Storage-Token'] = storage_token
    self.assertEqual(storage_token, self.request.user_token)
    self.request.headers['X-Auth-Token'] = user_token
    self.assertEqual(user_token, self.request.user_token)