from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_adapter_rest(self):
    api = 'private/cli/ucadmin'
    query = {'node': self.parameters['node_name'], 'adapter': self.parameters['adapter_name']}
    body = {}
    if self.parameters.get('type') is not None:
        body['type'] = self.parameters['type']
    if self.parameters.get('mode') is not None:
        body['mode'] = self.parameters['mode']
    dummy, error = rest_generic.patch_async(self.rest_api, api, None, body, query)
    if error:
        self.module.fail_json(msg='Error modifying adapter %s: %s' % (self.parameters['adapter_name'], to_native(error)))