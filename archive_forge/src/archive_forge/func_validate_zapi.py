from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import time
def validate_zapi(self, modify):
    if self.parameters['is_enabled'] is False:
        if len(modify) > 1 and 'is_enabled' in modify:
            self.module.fail_json(msg='Error: Cannot modify any other parameter for a service processor network if option "is_enabled" is set to false.')
        elif modify and 'is_enabled' not in modify:
            self.module.fail_json(msg='Error: Cannot modify a service processor network if it is disabled in ZAPI.')