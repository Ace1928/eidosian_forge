from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_cifs_local_group_rest(self):
    """
        Destroy the local group of an SVM.
        """
    api = 'protocols/cifs/local-groups/%s/%s' % (self.svm_uuid, self.sid)
    record, error = rest_generic.delete_async(self.rest_api, api, None)
    if error:
        self.module.fail_json(msg='Error on deleting cifs local-group: %s' % error)