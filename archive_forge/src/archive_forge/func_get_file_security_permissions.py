from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def get_file_security_permissions(self):
    api = 'protocols/file-security/permissions/%s/%s' % (self.svm_uuid, self.url_encode(self.parameters['path']))
    fields = 'acls,control_flags,group,owner'
    record, error = rest_generic.get_one_record(self.rest_api, api, {'fields': fields})
    if error:
        if '655865' in error and self.parameters['state'] == 'absent':
            return None
        self.module.fail_json(msg='Error fetching file security permissions %s: %s' % (self.parameters['path'], to_native(error)), exception=traceback.format_exc())
    return self.form_current(record) if record else None