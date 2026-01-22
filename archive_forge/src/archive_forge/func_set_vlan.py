from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.edgeswitch.edgeswitch import load_config, run_commands
from ansible_collections.community.network.plugins.module_utils.network.edgeswitch.edgeswitch import build_aggregate_spec, map_params_to_obj
from ansible_collections.community.network.plugins.module_utils.network.edgeswitch.edgeswitch_interface import InterfaceConfiguration, merge_interfaces
def set_vlan(self, vlan_id, type):
    try:
        self.tagged.remove(vlan_id)
    except ValueError:
        pass
    try:
        self.untagged.remove(vlan_id)
    except ValueError:
        pass
    try:
        self.excluded.remove(vlan_id)
    except ValueError:
        pass
    f = getattr(self, type)
    f.append(vlan_id)