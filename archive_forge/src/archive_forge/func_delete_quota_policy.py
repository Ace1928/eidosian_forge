from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import zapis_svm
def delete_quota_policy(self):
    """
        Deletes a quota policy
        """
    quota_policy_obj = netapp_utils.zapi.NaElement('quota-policy-delete')
    quota_policy_obj.add_new_child('policy-name', self.parameters['name'])
    try:
        self.server.invoke_successfully(quota_policy_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error deleting quota policy %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())