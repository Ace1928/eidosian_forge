from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
def get_single_afi_modify_requests(self, to_modify_afi):
    """build requests to modify a single afi family. Uses passed in config to find which family and what to change

        :param to_modify_afi: dictionary specifying the config to add/change in argspec format. expected to be for a single afi
        :param v: version number of afi to modify
        """
    requests = []
    v = self.afi_to_vnum(to_modify_afi)
    if to_modify_afi.get('enabled') is not None:
        payload = {'openconfig-dhcp-snooping:dhcpv{v}-admin-enable'.format(v=v): to_modify_afi['enabled']}
        uri = self.enable_uri.format(v=v)
        requests.append({'path': uri, 'method': self.patch_method_value, 'data': payload})
    if to_modify_afi.get('verify_mac') is not None:
        payload = {'openconfig-dhcp-snooping:dhcpv{v}-verify-mac-address'.format(v=v): to_modify_afi['verify_mac']}
        uri = self.verify_mac_uri.format(v=v)
        requests.append({'path': uri, 'method': self.patch_method_value, 'data': payload})
    if to_modify_afi.get('trusted'):
        for intf in to_modify_afi.get('trusted'):
            intf_name = intf.get('intf_name')
            if intf_name:
                payload = {'openconfig-interfaces:dhcpv{v}-snooping-trust'.format(v=v): 'ENABLE'}
                uri = self.trusted_uri.format(name=intf_name, v=v)
                requests.append({'path': uri, 'method': self.patch_method_value, 'data': payload})
    if to_modify_afi.get('vlans'):
        for vlan_id in to_modify_afi.get('vlans'):
            payload = {'sonic-vlan:dhcpv{v}_snooping_enable'.format(v=v): 'enable'}
            uri = self.vlans_uri.format(vlan_name='Vlan' + vlan_id, v=v)
            requests.append({'path': uri, 'method': self.patch_method_value, 'data': payload})
    if to_modify_afi.get('source_bindings'):
        entries = []
        for entry in to_modify_afi.get('source_bindings'):
            if entry.get('mac_addr'):
                entries.append({'mac': entry.get('mac_addr'), 'iptype': 'ipv' + str(v), 'config': {'mac': entry.get('mac_addr'), 'iptype': 'ipv' + str(v), 'vlan': 'Vlan' + str(entry.get('vlan_id')), 'interface': entry.get('intf_name'), 'ip': entry.get('ip_addr')}})
        payload = {'openconfig-dhcp-snooping:entry': entries}
        uri = self.binding_uri
        requests.append({'path': uri, 'method': self.patch_method_value, 'data': payload})
    return requests