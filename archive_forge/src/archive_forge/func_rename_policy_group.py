from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def rename_policy_group(self):
    """
        Rename policy group name.
        """
    rename_obj = netapp_utils.zapi.NaElement('qos-adaptive-policy-group-rename')
    rename_obj.add_new_child('new-name', self.parameters['name'])
    rename_obj.add_new_child('policy-group-name', self.parameters['from_name'])
    try:
        self.server.invoke_successfully(rename_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error renaming adaptive qos policy group %s: %s' % (self.parameters['from_name'], to_native(error)), exception=traceback.format_exc())