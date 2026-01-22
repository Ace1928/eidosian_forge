from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
def test_should_get_secret_acl_with_extra_trailing_slashes(self):
    self.responses.get(requests_mock.ANY, json=self.get_acl_response_data())
    self.manager.get(entity_ref=self.secret_ref + '///')
    self.assertEqual(self.secret_acl_ref, self.responses.last_request.url)