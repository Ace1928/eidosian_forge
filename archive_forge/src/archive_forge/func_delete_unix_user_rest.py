from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_unix_user_rest(self, current):
    """
        Deletes a UNIX user configuration for the specified SVM with rest API.
        """
    if not self.use_rest:
        return self.delete_unix_user()
    api = 'name-services/unix-users/%s' % current['svm']['uuid']
    dummy, error = rest_generic.delete_async(self.rest_api, api, self.parameters['name'])
    if error is not None:
        self.module.fail_json(msg='Error on deleting unix-user: %s' % error)