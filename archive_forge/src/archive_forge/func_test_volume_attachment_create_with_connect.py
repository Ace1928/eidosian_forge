from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_attachment
def test_volume_attachment_create_with_connect(self):
    self.volume_client.api_version = api_versions.APIVersion('3.54')
    arglist = [self.volume.id, self.server.id, '--connect', '--initiator', 'iqn.1993-08.org.debian:01:cad181614cec', '--ip', '192.168.1.20', '--host', 'my-host', '--platform', 'x86_64', '--os-type', 'linux2', '--multipath', '--mountpoint', '/dev/vdb', '--mode', 'null']
    verifylist = [('volume', self.volume.id), ('server', self.server.id), ('connect', True), ('initiator', 'iqn.1993-08.org.debian:01:cad181614cec'), ('ip', '192.168.1.20'), ('host', 'my-host'), ('platform', 'x86_64'), ('os_type', 'linux2'), ('multipath', True), ('mountpoint', '/dev/vdb'), ('mode', 'null')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    connect_info = dict([('initiator', 'iqn.1993-08.org.debian:01:cad181614cec'), ('ip', '192.168.1.20'), ('host', 'my-host'), ('platform', 'x86_64'), ('os_type', 'linux2'), ('multipath', True), ('mountpoint', '/dev/vdb')])
    self.volumes_mock.get.assert_called_once_with(self.volume.id)
    self.servers_mock.get.assert_called_once_with(self.server.id)
    self.volume_attachments_mock.create.assert_called_once_with(self.volume.id, connect_info, self.server.id, 'null')
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)