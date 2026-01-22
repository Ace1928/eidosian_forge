from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_nfs_service_rest(self):
    if self.svm_uuid is None:
        self.module.fail_json(msg='Error deleting nfs service for SVM %s: svm.uuid is None' % self.parameters['vserver'])
    dummy, error = rest_generic.delete_async(self.rest_api, 'protocols/nfs/services', self.svm_uuid, job_timeout=120)
    if error:
        self.module.fail_json(msg='Error deleting nfs service for SVM %s' % self.parameters['vserver'])