from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_role_privilege(self, privilege):
    api = 'security/roles/%s/%s/privileges' % (self.owner_uuid, self.parameters['name'])
    body = {'path': privilege['path'], 'access': privilege['access']}
    dummy, error = rest_generic.post_async(self.rest_api, api, body, job_timeout=120)
    if error:
        self.module.fail_json(msg='Error creating role privilege %s: %s' % (privilege['path'], to_native(error)), exception=traceback.format_exc())