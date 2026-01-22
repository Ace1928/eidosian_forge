from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def zapi_errors(self):
    unsupported_zapi_properties = ['consistency_group', 'comment']
    used_unsupported_zapi_properties = [option for option in unsupported_zapi_properties if option in self.parameters]
    if used_unsupported_zapi_properties:
        self.module.fail_json(msg='Error: %s options supported only with REST.' % ' ,'.join(used_unsupported_zapi_properties))
    if self.parameters.get('volumes') is None:
        self.module.fail_json(msg="Error: 'volumes' option is mandatory while using ZAPI.")
    if self.parameters.get('state') == 'absent':
        self.module.fail_json(msg='Deletion of consistency group snapshot is not supported with ZAPI.')