from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def validate_efficiency_compression(self, modify):
    """
        validate:
          - no efficiency keys are set when state is disabled.
        """
    if self.parameters['enabled'] == 'disabled':
        unsupported_enable_eff_keys = ['enable_compression', 'enable_inline_compression', 'enable_inline_dedupe', 'enable_cross_volume_inline_dedupe', 'enable_cross_volume_background_dedupe', 'enable_data_compaction']
        used_unsupported_enable_eff_keys = [key for key in unsupported_enable_eff_keys if self.parameters.get(key)]
        if used_unsupported_enable_eff_keys:
            disable_str = 'when volume efficiency already disabled, retry with state: present'
            if modify.get('enabled') == 'disabled':
                disable_str = 'when trying to disable volume efficiency'
            self.module.fail_json(msg='Error: cannot set compression keys: %s %s' % (used_unsupported_enable_eff_keys, disable_str))