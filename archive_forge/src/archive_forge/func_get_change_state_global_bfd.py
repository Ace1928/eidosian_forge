from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def get_change_state_global_bfd(self):
    """get ipv4 global bfd change state"""
    self.get_global_bfd(self.state)
    change = False
    if self.state == 'present':
        if self.static_routes_info['sroute_global_bfd']:
            for static_route in self.static_routes_info['sroute_global_bfd']:
                if static_route is not None:
                    if self.min_tx_interval is not None:
                        if int(static_route['minTxInterval']) != self.min_tx_interval:
                            change = True
                    if self.min_rx_interval is not None:
                        if int(static_route['minRxInterval']) != self.min_rx_interval:
                            change = True
                    if self.detect_multiplier is not None:
                        if int(static_route['multiplier']) != self.detect_multiplier:
                            change = True
                    return change
                else:
                    continue
        else:
            change = True
    else:
        if self.commands:
            if self.static_routes_info['sroute_global_bfd']:
                for static_route in self.static_routes_info['sroute_global_bfd']:
                    if static_route is not None:
                        if int(static_route['minTxInterval']) != 1000 or int(static_route['minRxInterval']) != 1000 or int(static_route['multiplier']) != 3:
                            change = True
        return change