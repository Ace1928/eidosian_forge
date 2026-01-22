from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_modify_specific_lldp_global_param_requests(self, command):
    """Get requests to modify specific LLDP Global configurations
        based on the command specified for the interface
        """
    requests = []
    if not command:
        return requests
    if 'enable' in command and command['enable'] is not None:
        payload = {'openconfig-lldp:enabled': command['enable']}
        url = self.lldp_global_config_path['enable']
        requests.append({'path': url, 'method': PATCH, 'data': payload})
    if 'hello_time' in command and command['hello_time'] is not None:
        payload = {'openconfig-lldp:hello-timer': str(command['hello_time'])}
        url = self.lldp_global_config_path['hello_time']
        requests.append({'path': url, 'method': PATCH, 'data': payload})
    if 'mode' in command and command['mode'] is not None:
        payload = {'openconfig-lldp-ext:mode': command['mode'].upper()}
        url = self.lldp_global_config_path['mode']
        requests.append({'path': url, 'method': PATCH, 'data': payload})
    if 'multiplier' in command and command['multiplier'] is not None:
        payload = {'openconfig-lldp-ext:multiplier': int(command['multiplier'])}
        url = self.lldp_global_config_path['multiplier']
        requests.append({'path': url, 'method': PATCH, 'data': payload})
    if 'system_name' in command and command['system_name'] is not None:
        payload = {'openconfig-lldp:system-name': command['system_name']}
        url = self.lldp_global_config_path['system_name']
        requests.append({'path': url, 'method': PATCH, 'data': payload})
    if 'system_description' in command and command['system_description'] is not None:
        payload = {'openconfig-lldp:system-description': command['system_description']}
        url = self.lldp_global_config_path['system_description']
        requests.append({'path': url, 'method': PATCH, 'data': payload})
    if 'tlv_select' in command:
        if 'management_address' in command['tlv_select']:
            payload = {'openconfig-lldp:suppress-tlv-advertisement': ['MANAGEMENT_ADDRESS']}
            url = self.lldp_global_config_path['tlv_select']
            if command['tlv_select']['management_address'] is False:
                requests.append({'path': url, 'method': PATCH, 'data': payload})
            elif command['tlv_select']['management_address'] is True:
                url = self.lldp_suppress_tlv.format(lldp_suppress_tlv='MANAGEMENT_ADDRESS')
                requests.append({'path': url, 'method': DELETE})
        if 'system_capabilities' in command['tlv_select']:
            payload = {'openconfig-lldp:suppress-tlv-advertisement': ['SYSTEM_CAPABILITIES']}
            url = self.lldp_global_config_path['tlv_select']
            if command['tlv_select']['system_capabilities'] is False:
                requests.append({'path': url, 'method': PATCH, 'data': payload})
            elif command['tlv_select']['system_capabilities'] is True:
                url = self.lldp_suppress_tlv.format(lldp_suppress_tlv='SYSTEM_CAPABILITIES')
                requests.append({'path': url, 'method': DELETE})
    return requests