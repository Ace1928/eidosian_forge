from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_ems_config_rest(self, modify):
    """Modify EMS config"""
    dummy, error = rest_generic.patch_async(self.rest_api, 'support/ems', None, modify)
    if error:
        self.module.fail_json(msg='Error modifying EMS config: %s.' % to_native(error), exception=traceback.format_exc())