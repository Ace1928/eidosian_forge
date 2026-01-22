from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_subnet_rest(self, modify):
    api = 'network/ip/subnets'
    dummy, error = rest_generic.patch_async(self.rest_api, api, self.uuid, self.form_create_modify_body_rest(modify))
    if error:
        self.module.fail_json(msg='Error modifying subnet %s: %s' % (self.parameters.get('name'), to_native(error)))