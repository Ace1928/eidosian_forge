from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def validate_module_attributes(self):
    """
        Validate module attributes
        :return None
        """
    param_list = ['nfs_export_name', 'nfs_export_id', 'filesystem_name', 'filesystem_id', 'nas_server_id', 'snapshot_name', 'snapshot_id', 'path']
    for param in param_list:
        if self.module.params[param] and len(self.module.params[param].strip()) == 0:
            msg = 'Please provide valid value for: %s' % param
            LOG.error(msg)
            self.module.fail_json(msg=msg)