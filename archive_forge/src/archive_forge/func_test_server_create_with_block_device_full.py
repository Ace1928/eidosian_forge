import collections
import copy
import getpass
import json
import tempfile
from unittest import mock
from unittest.mock import call
import iso8601
from novaclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
def test_server_create_with_block_device_full(self):
    self.compute_client.api_version = api_versions.APIVersion('2.67')
    block_device = f'uuid={self.volume.id},source_type=volume,destination_type=volume,disk_bus=ide,device_type=disk,device_name=sdb,guest_format=ext4,volume_size=64,volume_type=foo,boot_index=1,delete_on_termination=true,tag=foo'
    block_device_alt = f'uuid={self.volume_alt.id},source_type=volume'
    arglist = ['--image', 'image1', '--flavor', self.flavor.id, '--block-device', block_device, '--block-device', block_device_alt, self.new_server.name]
    verifylist = [('image', 'image1'), ('flavor', self.flavor.id), ('block_devices', [{'uuid': self.volume.id, 'source_type': 'volume', 'destination_type': 'volume', 'disk_bus': 'ide', 'device_type': 'disk', 'device_name': 'sdb', 'guest_format': 'ext4', 'volume_size': '64', 'volume_type': 'foo', 'boot_index': '1', 'delete_on_termination': 'true', 'tag': 'foo'}, {'uuid': self.volume_alt.id, 'source_type': 'volume'}]), ('server_name', self.new_server.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'meta': None, 'files': {}, 'reservation_id': None, 'min_count': 1, 'max_count': 1, 'security_groups': [], 'userdata': None, 'key_name': None, 'availability_zone': None, 'admin_pass': None, 'block_device_mapping_v2': [{'uuid': self.volume.id, 'source_type': 'volume', 'destination_type': 'volume', 'disk_bus': 'ide', 'device_name': 'sdb', 'volume_size': '64', 'guest_format': 'ext4', 'boot_index': 1, 'device_type': 'disk', 'delete_on_termination': True, 'tag': 'foo', 'volume_type': 'foo'}, {'uuid': self.volume_alt.id, 'source_type': 'volume', 'destination_type': 'volume'}], 'nics': 'auto', 'scheduler_hints': {}, 'config_drive': None}
    self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist(), data)