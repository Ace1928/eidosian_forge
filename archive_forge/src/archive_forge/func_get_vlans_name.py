from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, execute_nc_action, ce_argument_spec
def get_vlans_name(self):
    """ get all vlan vid and its name  list,
        sample: [ ("20", "VLAN_NAME_20"), ("30", "VLAN_NAME_30") ]"""
    conf_str = CE_NC_GET_VLANS
    xml_str = get_nc_config(self.module, conf_str)
    vlan_list = list()
    if '<data/>' in xml_str:
        return vlan_list
    else:
        vlan_list = re.findall('.*<vlanId>(.*)</vlanId>.*\\s*<vlanName>(.*)</vlanName>.*', xml_str)
        return vlan_list