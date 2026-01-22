from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def cifs_local_set_passwd_rest(self):
    self.get_svm_uuid()
    sid = self.get_user_sid()
    api = 'protocols/cifs/local-users'
    uuids = '%s/%s' % (self.svm_uuid, sid)
    body = {'password': self.parameters['user_password']}
    dummy, error = rest_generic.patch_async(self.rest_api, api, uuids, body, job_timeout=120)
    if error:
        self.module.fail_json(msg='Error change password for user %s: %s' % (self.parameters['user_name'], to_native(error)), exception=traceback.format_exc())