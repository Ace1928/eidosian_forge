from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, execute_nc_action, ce_argument_spec
def get_vlan_attr(self, vlan_id):
    """ get vlan attributes."""
    conf_str = CE_NC_GET_VLAN % vlan_id
    xml_str = get_nc_config(self.module, conf_str)
    attr = dict()
    if '<data/>' in xml_str:
        return attr
    else:
        re_find_id = re.findall('.*<vlanId>(.*)</vlanId>.*\\s*', xml_str)
        re_find_name = re.findall('.*<vlanName>(.*)</vlanName>.*\\s*', xml_str)
        re_find_desc = re.findall('.*<vlanDesc>(.*)</vlanDesc>.*\\s*', xml_str)
        if re_find_id:
            if re_find_name:
                attr = dict(vlan_id=re_find_id[0], name=re_find_name[0], description=re_find_desc[0])
            else:
                attr = dict(vlan_id=re_find_id[0], name=None, description=re_find_desc[0])
        return attr