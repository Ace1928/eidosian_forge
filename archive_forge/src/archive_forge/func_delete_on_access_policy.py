from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def delete_on_access_policy(self):
    """
        Delete a Vscan On Access Policy
        :return:
        """
    if self.use_rest:
        return self.delete_on_access_policy_rest()
    access_policy_obj = netapp_utils.zapi.NaElement('vscan-on-access-policy-delete')
    access_policy_obj.add_new_child('policy-name', self.parameters['policy_name'])
    try:
        self.server.invoke_successfully(access_policy_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error Deleting Vscan on Access Policy %s: %s' % (self.parameters['policy_name'], to_native(error)), exception=traceback.format_exc())