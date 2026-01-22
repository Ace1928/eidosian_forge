from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def delete_cifs_local_user(self):
    api = 'protocols/cifs/local-users'
    uuids = '%s/%s' % (self.svm_uuid, self.sid)
    dummy, error = rest_generic.delete_async(self.rest_api, api, uuids)
    if error:
        self.module.fail_json(msg='Error while deleting CIFS local user: %s' % error)