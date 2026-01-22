from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def set_sd(self):
    set_sd = netapp_utils.zapi.NaElement('file-directory-security-set')
    set_sd.add_new_child('policy-name', self.parameters['policy_name'])
    if self.parameters.get('ignore-broken-symlinks'):
        set_sd.add_new_child('ignore-broken-symlinks', str(self.parameters['ignore_broken_symlinks']))
    try:
        self.server.invoke_successfully(set_sd, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error applying file-directory policy %s: %s' % (self.parameters['policy_name'], to_native(error)), exception=traceback.format_exc())