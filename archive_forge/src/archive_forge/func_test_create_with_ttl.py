from unittest.mock import patch
import uuid
import testtools
from designateclient import exceptions
from designateclient.tests import v2
from designateclient.v2 import zones
def test_create_with_ttl(self):
    ref = self.new_ref(ttl=60)
    parts = ['zones', ZONE['id'], self.RESOURCE]
    self.stub_url('POST', parts=parts, json=ref)
    values = ref.copy()
    del values['id']
    self.client.recordsets.create(ZONE['id'], '{}.{}'.format(values['name'], ZONE['name']), values['type'], values['records'], ttl=values['ttl'])
    values['name'] = '{}.{}'.format(ref['name'], ZONE['name'])
    self.assertRequestBodyIs(json=values)