from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \

        Modify the snapshot policy attributes
        :param snap_pol_details: Details of the snapshot policy
        :param modify_dict: Dictionary containing the attributes of
         snapshot policy which are to be updated
        :return: True, if the operation is successful
        