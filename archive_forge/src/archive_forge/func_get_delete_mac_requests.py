from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
def get_delete_mac_requests(self, commands, have, is_delete_all):
    requests = []
    for cmd in commands:
        vrf_name = cmd.get('vrf_name', None)
        if vrf_name and is_delete_all:
            requests.extend(self.get_delete_all_mac_requests(vrf_name))
        else:
            mac = cmd.get('mac', {})
            if mac:
                aging_time = mac.get('aging_time', None)
                dampening_interval = mac.get('dampening_interval', None)
                dampening_threshold = mac.get('dampening_threshold', None)
                mac_table_entries = mac.get('mac_table_entries', [])
                for cfg in have:
                    cfg_vrf_name = cfg.get('vrf_name', None)
                    cfg_mac = cfg.get('mac', {})
                    if cfg_mac:
                        cfg_aging_time = cfg_mac.get('aging_time', None)
                        cfg_dampening_interval = cfg_mac.get('dampening_interval', None)
                        cfg_dampening_threshold = cfg_mac.get('dampening_threshold', None)
                        cfg_mac_table_entries = cfg_mac.get('mac_table_entries', [])
                        if vrf_name and vrf_name == cfg_vrf_name:
                            if aging_time and aging_time == cfg_aging_time:
                                requests.append(self.get_delete_fdb_cfg_attr(vrf_name, 'mac-aging-time'))
                            if dampening_interval and dampening_interval == cfg_dampening_interval:
                                requests.append(self.get_delete_mac_dampening_attr(vrf_name, 'interval'))
                            if dampening_threshold and dampening_threshold == cfg_dampening_threshold:
                                requests.append(self.get_delete_mac_dampening_attr(vrf_name, 'threshold'))
                            if mac_table_entries:
                                for entry in mac_table_entries:
                                    mac_address = entry.get('mac_address', None)
                                    vlan_id = entry.get('vlan_id', None)
                                    interface = entry.get('interface', None)
                                    if cfg_mac_table_entries:
                                        for cfg_entry in cfg_mac_table_entries:
                                            cfg_mac_address = cfg_entry.get('mac_address', None)
                                            cfg_vlan_id = cfg_entry.get('vlan_id', None)
                                            cfg_interface = cfg_entry.get('interface', None)
                                            if mac_address and vlan_id and (mac_address == cfg_mac_address) and (vlan_id == cfg_vlan_id):
                                                if interface and interface == cfg_interface:
                                                    requests.append(self.get_delete_mac_table_intf(vrf_name, mac_address, vlan_id))
                                                elif not interface:
                                                    requests.append(self.get_delete_mac_table_entry(vrf_name, mac_address, vlan_id))
    return requests