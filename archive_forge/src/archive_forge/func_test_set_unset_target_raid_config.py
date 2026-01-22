import json
import ddt
from tempest.lib.common.utils import data_utils
from ironicclient.tests.functional.osc.v1 import base
@ddt.data((50, '1'), ('MAX', 'JBOD'), (300, '6+0'))
@ddt.unpack
def test_set_unset_target_raid_config(self, size, raid_level):
    """Set and unset node target RAID config data.

        Test steps:
        1) Create baremetal node in setUp.
        2) Set target RAID config data for the node
        3) Check target_raid_config of node equals to expected value.
        4) Unset target_raid_config data.
        5) Check that target_raid_config data is empty.
        """
    min_version = '--os-baremetal-api-version 1.12'
    argument_json = {'logical_disks': [{'size_gb': size, 'raid_level': raid_level}]}
    argument_string = json.dumps(argument_json)
    self.openstack("baremetal node set --target-raid-config '{}' {} {}".format(argument_string, self.node['uuid'], min_version))
    show_prop = self.node_show(self.node['uuid'], ['target_raid_config'], min_version)
    self.assert_dict_is_subset(argument_json, show_prop['target_raid_config'])
    self.openstack('baremetal node unset --target-raid-config {} {}'.format(self.node['uuid'], min_version))
    show_prop = self.node_show(self.node['uuid'], ['target_raid_config'], min_version)
    self.assertEqual({}, show_prop['target_raid_config'])