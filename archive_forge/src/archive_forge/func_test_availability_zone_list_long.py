from openstackclient.common import availability_zone
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
def test_availability_zone_list_long(self):
    arglist = ['--long']
    verifylist = [('long', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.availability_zones.assert_called_with(details=True)
    self.volume_sdk_client.availability_zones.assert_called_with()
    self.network_client.availability_zones.assert_called_with()
    self.assertEqual(self.long_columnslist, columns)
    datalist = ()
    for compute_az in self.compute_azs:
        datalist += _build_compute_az_datalist(compute_az, long_datalist=True)
    for volume_az in self.volume_azs:
        datalist += _build_volume_az_datalist(volume_az, long_datalist=True)
    for network_az in self.network_azs:
        datalist += _build_network_az_datalist(network_az, long_datalist=True)
    self.assertEqual(datalist, tuple(data))