from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def remove_task_from_policy(self):
    policy_remove_task = netapp_utils.zapi.NaElement('file-directory-security-policy-task-remove')
    policy_remove_task.add_new_child('path', self.parameters['path'])
    policy_remove_task.add_new_child('policy-name', self.parameters['policy_name'])
    try:
        self.server.invoke_successfully(policy_remove_task, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error removing task from file-directory policy %s: %s' % (self.parameters['policy_name'], to_native(error)), exception=traceback.format_exc())