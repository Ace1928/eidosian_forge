from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_name_mappings_rest(self, modify=None, reindex=False):
    """
        Updates the name mapping configuration of an SVM with rest API.
        Swap the position with new position(new_index).
        """
    body = {}
    query = None
    if modify:
        for option in ['pattern', 'replacement', 'client_match']:
            if option in modify:
                body[option] = self.parameters[option]
    index = self.parameters['index']
    if reindex:
        query = {'new_index': self.parameters.get('index')}
        index = self.parameters['from_index']
    api = 'name-services/name-mappings/%s/%s/%s' % (self.svm_uuid, self.parameters['direction'], index)
    dummy, error = rest_generic.patch_async(self.rest_api, api, None, body, query)
    if error is not None:
        self.module.fail_json(msg='Error on modifying name mappings rest: %s' % error)