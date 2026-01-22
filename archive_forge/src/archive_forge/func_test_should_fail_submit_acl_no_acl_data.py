from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
def test_should_fail_submit_acl_no_acl_data(self):
    data = {'acl_ref': self.secret_acl_ref}
    self.responses.put(self.secret_acl_ref, json=data)
    entity = self.manager.create(entity_ref=self.secret_ref + '///')
    self.assertRaises(ValueError, entity.submit)