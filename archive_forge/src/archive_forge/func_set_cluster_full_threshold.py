from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_module import NetAppModule
def set_cluster_full_threshold(self, stage2_aware_threshold=None, stage3_block_threshold_percent=None, max_metadata_over_provision_factor=None):
    """
        modify cluster full threshold
        """
    try:
        self.sfe.modify_cluster_full_threshold(stage2_aware_threshold=stage2_aware_threshold, stage3_block_threshold_percent=stage3_block_threshold_percent, max_metadata_over_provision_factor=max_metadata_over_provision_factor)
    except Exception as exception_object:
        self.module.fail_json(msg='Failed to modify cluster full threshold %s' % to_native(exception_object), exception=traceback.format_exc())