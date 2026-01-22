import time
import uuid
from designateclient.tests import v2
def test_delete_zone_share(self):
    ref = self.new_ref()
    parts = ['zones', self.zone_id, 'shares', ref['id']]
    self.stub_url('DELETE', parts=parts)
    response = self.client.zone_share.delete(self.zone_id, ref['id'])
    self.assertRequestBodyIs(None)
    self.assertEqual('', response)