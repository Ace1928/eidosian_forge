import datetime
import uuid
from oslo_utils import timeutils
from keystonemiddleware import fixture
from keystonemiddleware.tests.unit.auth_token import test_auth_token_middleware
def test_auth_token_fixture_valid_token(self):
    resp = self.call_middleware(headers={'X-Auth-Token': self.token_id})
    self.assertIn('keystone.token_info', resp.request.environ)