from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def get_dacl(self):
    dacl_entry = None
    advanced_access_list = None
    dacl_get_iter = netapp_utils.zapi.NaElement('file-directory-security-ntfs-dacl-get-iter')
    dacl_info = netapp_utils.zapi.NaElement('file-directory-security-ntfs-dacl')
    dacl_info.add_new_child('vserver', self.parameters['vserver'])
    dacl_info.add_new_child('ntfs-sd', self.parameters['security_descriptor'])
    dacl_info.add_new_child('access-type', self.parameters['access_type'])
    dacl_info.add_new_child('account', self.parameters['account'])
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(dacl_info)
    dacl_get_iter.add_child_elem(query)
    try:
        result = self.server.invoke_successfully(dacl_get_iter, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching %s DACL for account %s for security descriptor %s: %s' % (self.parameters['access_type'], self.parameters['account'], self.parameters['security_descriptor'], to_native(error)), exception=traceback.format_exc())
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
        attributes_list = result.get_child_by_name('attributes-list')
        if attributes_list is None:
            return None
        dacl = attributes_list.get_child_by_name('file-directory-security-ntfs-dacl')
        apply_to_list = []
        apply_to = dacl.get_child_by_name('apply-to')
        for apply_child in apply_to.get_children():
            inheritance_level = apply_child.get_content()
            apply_to_list.append(inheritance_level)
        if dacl.get_child_by_name('advanced-rights'):
            advanced_access_list = []
            advanced_access = dacl.get_child_by_name('advanced-rights')
            for right in advanced_access.get_children():
                advanced_access_right = right.get_content()
                advanced_right = {'advanced_access_rights': advanced_access_right}
                advanced_access_list.append(advanced_right)
        dacl_entry = {'access_type': dacl.get_child_content('access-type'), 'account': dacl.get_child_content('account'), 'apply_to': apply_to_list, 'security_descriptor': dacl.get_child_content('ntfs-sd'), 'readable_access_rights': dacl.get_child_content('readable-access-rights'), 'vserver': dacl.get_child_content('vserver')}
        if advanced_access_list is not None:
            dacl_entry['advanced_rights'] = advanced_access_list
        else:
            dacl_entry['rights'] = dacl.get_child_content('rights')
    return dacl_entry