from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def modify_s3_groups(self, modify):
    api = 'protocols/s3/services/%s/groups' % self.svm_uuid
    body = {}
    if modify.get('comment') is not None:
        body['comment'] = self.parameters['comment']
    if modify.get('users') is not None:
        body['users'] = self.parameters['users']
    if modify.get('policies') is not None:
        body['policies'] = self.parameters['policies']
    dummy, error = rest_generic.patch_async(self.rest_api, api, self.group_id, body)
    if error:
        self.module.fail_json(msg='Error modifying S3 group %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())