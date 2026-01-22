import time
import uuid
from designateclient.tests import v2
def test_get_accept(self):
    accept_id = '098bee04-fe30-4a83-8ccd-e0c496755816'
    ref = {'status': 'COMPLETE'}
    parts = ['zones', 'tasks', 'transfer_accepts', accept_id]
    self.stub_url('GET', parts=parts, json=ref)
    response = self.client.zone_transfers.get_accept(accept_id)
    self.assertEqual(ref, response)