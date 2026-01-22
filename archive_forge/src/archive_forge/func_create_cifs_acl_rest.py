from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_cifs_acl_rest(self, svm_uuid):
    """
        create CIFS share acl with rest API.
        """
    if not self.use_rest:
        return self.create_cifs_acl()
    body = {'user_or_group': self.parameters.get('user_or_group'), 'permission': self.parameters.get('permission')}
    ug_type = self.parameters.get('type')
    if ug_type:
        body['type'] = ug_type
    api = 'protocols/cifs/shares/%s/%s/acls' % (svm_uuid['uuid'], self.parameters.get('share_name'))
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error is not None:
        self.module.fail_json(msg='Error on creating cifs share acl: %s' % error)