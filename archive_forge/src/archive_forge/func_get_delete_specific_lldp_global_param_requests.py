from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_delete_specific_lldp_global_param_requests(self, command, config):
    """Get requests to delete specific LLDP global configurations
        based on the command specified for the interface
        """
    requests = []
    if not command:
        return requests
    if 'hello_time' in command:
        url = self.lldp_global_config_path['hello_time']
        requests.append({'path': url, 'method': DELETE})
    if 'enable' in command:
        url = self.lldp_global_config_path['enable']
        if command['enable'] is False:
            payload = {'openconfig-lldp:enabled': True}
        elif command['enable'] is True:
            payload = {'openconfig-lldp:enabled': False}
        requests.append({'path': url, 'method': PATCH, 'data': payload})
    if 'mode' in command:
        url = self.lldp_global_config_path['mode']
        requests.append({'path': url, 'method': DELETE})
    if 'multiplier' in command:
        url = self.lldp_global_config_path['multiplier']
        requests.append({'path': url, 'method': DELETE})
    if 'system_name' in command:
        url = self.lldp_global_config_path['system_name']
        requests.append({'path': url, 'method': DELETE})
    if 'system_description' in command:
        url = self.lldp_global_config_path['system_description']
        requests.append({'path': url, 'method': DELETE})
    if 'tlv_select' in command:
        if 'management_address' in command['tlv_select']:
            payload = {'openconfig-lldp:suppress-tlv-advertisement': ['MANAGEMENT_ADDRESS']}
            url = self.lldp_global_config_path['tlv_select']
            if command['tlv_select']['management_address'] is True:
                requests.append({'path': url, 'method': PATCH, 'data': payload})
            elif command['tlv_select']['management_address'] is False:
                url = self.lldp_suppress_tlv.format(lldp_suppress_tlv='MANAGEMENT_ADDRESS')
                requests.append({'path': url, 'method': DELETE})
        if 'system_capabilities' in command['tlv_select']:
            payload = {'openconfig-lldp:suppress-tlv-advertisement': ['SYSTEM_CAPABILITIES']}
            url = self.lldp_global_config_path['tlv_select']
            if command['tlv_select']['system_capabilities'] is True:
                requests.append({'path': url, 'method': PATCH, 'data': payload})
            elif command['tlv_select']['system_capabilities'] is False:
                url = self.lldp_suppress_tlv.format(lldp_suppress_tlv='SYSTEM_CAPABILITIES')
                requests.append({'path': url, 'method': DELETE})
    return requests