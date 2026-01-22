from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_symlink_mapping_rest(self):
    """
        Removes a specific UNIX symbolink mapping for a SVM
        """
    api = 'protocols/cifs/unix-symlink-mapping/%s/%s' % (self.svm_uuid, self.encode_path(self.parameters['unix_path']))
    dummy, error = rest_generic.delete_async(self.rest_api, api, uuid=None)
    if error is not None:
        self.module.fail_json(msg='Error while deleting cifs unix symlink mapping: %s' % to_native(error))