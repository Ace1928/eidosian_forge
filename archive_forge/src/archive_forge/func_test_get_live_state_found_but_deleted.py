from unittest import mock
from heat.common import exception
from heat.engine.clients.os import cinder as c_plugin
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_get_live_state_found_but_deleted(self):
    self.my_encrypted_vol_type.resource_id = '1234'
    value = mock.MagicMock(spec=[])
    self.volume_encryption_types.get.return_value = value
    self.assertRaises(exception.EntityNotFound, self.my_encrypted_vol_type.get_live_state, self.my_encrypted_vol_type.properties)