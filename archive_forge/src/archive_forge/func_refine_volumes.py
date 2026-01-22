from __future__ import absolute_import, division, print_function
import logging
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def refine_volumes(self, volumes):
    """Refine volumes.
            :param volumes: Volumes that is to be added/removed
            :return: List of volumes with each volume being identified with either
            vol_id or vol_name
        """
    for vol in volumes:
        if vol['vol_id'] is not None and vol['vol_name'] is None:
            del vol['vol_name']
        elif vol['vol_name'] is not None and vol['vol_id'] is None:
            del vol['vol_id']
    return volumes