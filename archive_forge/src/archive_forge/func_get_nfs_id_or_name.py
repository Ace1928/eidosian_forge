from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_nfs_id_or_name(self):
    """ Provide nfs_export_id or nfs_export_name user given value

        :return: value provided by user in nfs_export_id/nfs_export_name
        :rtype: str
        """
    if self.module.params['nfs_export_id']:
        return self.module.params['nfs_export_id']
    return self.module.params['nfs_export_name']