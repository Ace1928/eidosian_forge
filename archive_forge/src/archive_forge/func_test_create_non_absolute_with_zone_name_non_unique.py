from unittest.mock import patch
import uuid
import testtools
from designateclient import exceptions
from designateclient.tests import v2
from designateclient.v2 import zones
@patch.object(zones.ZoneController, 'list')
def test_create_non_absolute_with_zone_name_non_unique(self, zone_list):
    zone_list.return_value = [1, 2]
    ref = self.new_ref()
    values = ref.copy()
    del values['id']
    with testtools.ExpectedException(exceptions.NoUniqueMatch):
        self.client.recordsets.create(ZONE['name'], '{}.{}'.format(values['name'], ZONE['name']), values['type'], values['records'])