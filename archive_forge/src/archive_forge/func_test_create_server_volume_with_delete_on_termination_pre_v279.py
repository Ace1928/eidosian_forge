from unittest import mock
from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import volumes
def test_create_server_volume_with_delete_on_termination_pre_v279(self):
    self.cs.api_version = api_versions.APIVersion('2.78')
    ex = self.assertRaises(TypeError, self.cs.volumes.create_server_volume, '1234', volume_id='15e59938-07d5-11e1-90e3-e3dffe0c5983', delete_on_termination=True)
    self.assertIn('delete_on_termination', str(ex))