from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_key_manager_rest(self):
    api = 'security/key-managers'
    if self.uuid is None:
        query = {'scope': self.scope}
        if self.scope == 'svm':
            query['svm.name'] = self.parameters['svm']['name']
    else:
        query = None
    dummy, error = rest_generic.delete_async(self.rest_api, api, self.uuid, query)
    if error:
        resource = 'cluster' if self.parameters.get('vserver') is None else self.parameters['vserver']
        self.module.fail_json(msg='Error deleting key manager for %s: %s' % (resource, error))