from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_krbrealm_rest(self):
    api = 'protocols/nfs/kerberos/realms/%s' % self.svm_uuid
    dummy, error = rest_generic.delete_async(self.rest_api, api, self.parameters['realm'])
    if error:
        self.module.fail_json(msg='Error deleting Kerberos Realm configuration %s: %s' % (self.parameters['realm'], to_native(error)))