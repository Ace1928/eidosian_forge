from tempest.lib.common.utils import data_utils
from ironicclient.tests.functional.osc.v1 import base
def test_llc_1_19(self):
    """Check baremetal port create command with LLC option.

        Test steps:
        1) Create port using --local-link-connection argument.
        2) Check that port successfully created with right LLC data.
        """
    fake_port_id = data_utils.rand_name(prefix='ovs-node-')
    fake_switch_id = data_utils.rand_mac_address()
    llc_value = {'switch_info': 'brbm', 'port_id': fake_port_id, 'switch_id': fake_switch_id}
    api_version = ' --os-baremetal-api-version 1.19'
    params = self.generate_params('--local-link-connection', llc_value)
    port = self.port_create(self.node['uuid'], params='{0} {1}'.format(params, api_version))
    self.assert_dict_is_subset(llc_value, port['local_link_connection'])