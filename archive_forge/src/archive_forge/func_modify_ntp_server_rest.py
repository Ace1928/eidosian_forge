from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_ntp_server_rest(self, modify):
    body = {}
    if modify.get('version'):
        body['version'] = modify['version']
    if modify.get('key_id'):
        body['key'] = {'id': modify['key_id']}
    if body:
        dummy, error = rest_generic.patch_async(self.rest_api, 'cluster/ntp/servers', self.parameters['server_name'], body)
        if error:
            self.module.fail_json(msg=error)