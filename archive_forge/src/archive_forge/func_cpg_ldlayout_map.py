from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.storage.hpe3par import hpe3par
def cpg_ldlayout_map(ldlayout_dict):
    if ldlayout_dict['RAIDType'] is not None and ldlayout_dict['RAIDType']:
        ldlayout_dict['RAIDType'] = client.HPE3ParClient.RAID_MAP[ldlayout_dict['RAIDType']]['raid_value']
    if ldlayout_dict['HA'] is not None and ldlayout_dict['HA']:
        ldlayout_dict['HA'] = getattr(client.HPE3ParClient, ldlayout_dict['HA'])
    return ldlayout_dict