from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_symlink_mapping_rest(self):
    """
        Creates a UNIX symbolink mapping for CIFS share
        """
    api = 'protocols/cifs/unix-symlink-mapping'
    body = {'svm.name': self.parameters['vserver'], 'unix_path': self.parameters['unix_path'], 'target': {'share': self.parameters['share_name'], 'path': self.parameters['cifs_path']}}
    if 'cifs_server' in self.parameters:
        body['target.server'] = self.parameters['cifs_server']
    if 'locality' in self.parameters:
        body['target.locality'] = self.parameters['locality']
    if 'home_directory' in self.parameters:
        body['target.home_directory'] = self.parameters['home_directory']
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error is not None:
        self.module.fail_json(msg='Error while creating cifs unix symlink mapping: %s' % to_native(error), exception=traceback.format_exc())