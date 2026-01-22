from openstack.object_store.v1 import account
from openstack.tests.unit import base
def test_set_account_temp_url_key_second(self):
    sot = account.Account()
    key = 'super-secure-key'
    self.register_uris([dict(method='POST', uri=self.endpoint, status_code=204, validate=dict(headers={'x-account-meta-temp-url-key-2': key})), dict(method='HEAD', uri=self.endpoint, headers={'x-account-meta-temp-url-key-2': key})])
    sot.set_temp_url_key(self.cloud.object_store, key, secondary=True)
    self.assert_calls()