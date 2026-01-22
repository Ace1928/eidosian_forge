from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_unix_group_rest(self, modify, current=None):
    """
        Updates UNIX group information for the specified user and SVM with rest API.
        """
    if not self.use_rest:
        return self.modify_unix_group(modify)
    if 'users' in modify:
        self.modify_users_in_group_rest(current)
        if len(modify) == 1:
            return
    api = 'name-services/unix-groups/%s' % current['svm']['uuid']
    body = {}
    if 'id' in modify:
        body['id'] = modify['id']
    if body:
        dummy, error = rest_generic.patch_async(self.rest_api, api, self.parameters['name'], body)
        if error is not None:
            self.module.fail_json(msg='Error on modifying UNIX group: %s' % error)