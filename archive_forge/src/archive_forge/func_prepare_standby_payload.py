from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
import copy
def prepare_standby_payload(standby_mdm):
    """prepare the payload for add standby MDM"""
    payload_dict = {}
    for mdm_keys in standby_mdm:
        if standby_mdm[mdm_keys]:
            payload_dict[mdm_keys] = standby_mdm[mdm_keys]
        else:
            payload_dict[mdm_keys] = None
    return payload_dict