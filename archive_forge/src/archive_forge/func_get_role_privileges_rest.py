from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_role_privileges_rest(self):
    api = 'security/roles/%s/%s/privileges' % (self.owner_uuid, self.parameters['name'])
    records, error = rest_generic.get_0_or_more_records(self.rest_api, api, {})
    if error:
        self.module.fail_json(msg='Error getting role privileges for role %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
    return self.format_privileges(records)