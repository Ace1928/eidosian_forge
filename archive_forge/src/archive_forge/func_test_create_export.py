import time
import uuid
from designateclient.tests import v2
def test_create_export(self):
    zone = uuid.uuid4().hex
    ref = {}
    parts = ['zones', zone, 'tasks', 'export']
    self.stub_url('POST', parts=parts, json=ref)
    self.client.zone_exports.create(zone)
    self.assertRequestBodyIs(json=ref)