import time
import uuid
from designateclient.tests import v2
def test_create_primary(self):
    ref = self.new_ref(email='root@example.com')
    self.stub_url('POST', parts=[self.RESOURCE], json=ref)
    values = ref.copy()
    del values['id']
    self.client.zones.create(values['name'], email=values['email'])
    self.assertRequestBodyIs(json=values)