from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def get_task_iter(self):
    task_get_iter = netapp_utils.zapi.NaElement('file-directory-security-policy-task-get-iter')
    task_info = netapp_utils.zapi.NaElement('file-directory-security-policy-task')
    task_info.add_new_child('vserver', self.parameters['vserver'])
    task_info.add_new_child('policy-name', self.parameters['policy_name'])
    task_info.add_new_child('path', self.parameters['path'])
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(task_info)
    task_get_iter.add_child_elem(query)
    try:
        result = self.server.invoke_successfully(task_get_iter, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching task from file-directory policy %s: %s' % (self.parameters['policy_name'], to_native(error)), exception=traceback.format_exc())
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
        attributes_list = result.get_child_by_name('attributes-list')
        task = attributes_list.get_child_by_name('file-directory-security-policy-task')
        task_result = dict()
        task_result['path'] = task.get_child_content('path')
        if task.get_child_by_name('ntfs-mode'):
            task_result['ntfs_mode'] = task.get_child_content('ntfs-mode')
        if task.get_child_by_name('security-type'):
            task_result['security_type'] = task.get_child_content('security-type')
        if task.get_child_by_name('ntfs-sd'):
            task_result['ntfs_sd'] = [ntfs_sd.get_content() for ntfs_sd in task.get_child_by_name('ntfs-sd').get_children()]
        return task_result
    return None