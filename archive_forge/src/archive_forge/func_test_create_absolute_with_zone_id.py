from unittest.mock import patch
import uuid
import testtools
from designateclient import exceptions
from designateclient.tests import v2
from designateclient.v2 import zones
def test_create_absolute_with_zone_id(self):
    ref = self.new_ref()
    parts = ['zones', ZONE['id'], self.RESOURCE]
    self.stub_url('POST', parts=parts, json=ref)
    values = ref.copy()
    del values['id']
    self.client.recordsets.create(ZONE['id'], '{}.{}'.format(values['name'], ZONE['name']), values['type'], values['records'])
    values['name'] = '{}.{}'.format(ref['name'], ZONE['name'])
    self.assertRequestBodyIs(json=values)