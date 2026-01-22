from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def create_body_or_query_common(self, params):
    result = {}
    if params.get('anonymous_user_id') is not None:
        result['anonymous_user'] = self.parameters['anonymous_user_id']
    if params.get('ntfs_unix_security') is not None:
        result['ntfs_unix_security'] = self.parameters['ntfs_unix_security']
    if params.get('allow_suid') is not None:
        result['allow_suid'] = self.parameters['allow_suid']
    if params.get('chown_mode') is not None:
        result['chown_mode'] = self.parameters['chown_mode']
    if params.get('allow_device_creation') is not None:
        result['allow_device_creation'] = self.parameters['allow_device_creation']
    return result