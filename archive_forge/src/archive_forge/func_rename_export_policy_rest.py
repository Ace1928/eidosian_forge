from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def rename_export_policy_rest(self, current):
    policy_id = current['id']
    params = {'name': self.parameters['name']}
    api = 'protocols/nfs/export-policies'
    dummy, error = rest_generic.patch_async(self.rest_api, api, policy_id, params)
    if error is not None:
        self.module.fail_json(msg='Error on renaming export policy: %s' % error)