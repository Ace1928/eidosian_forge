from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
@staticmethod
def get_vlan_ids_from_range(vlan_id_range):
    """Get start and end VLAN ID from VLAN ID range"""
    try:
        vlan_id_start, vlan_id_end = vlan_id_range.split('-')
    except (AttributeError, TypeError):
        vlan_id_start = vlan_id_end = vlan_id_range
    except ValueError:
        vlan_id_start = vlan_id_end = vlan_id_range.strip()
    return (vlan_id_start, vlan_id_end)