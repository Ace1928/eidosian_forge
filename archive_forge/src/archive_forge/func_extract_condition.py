from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def extract_condition(self, name, results):
    """ check if any of the conditions is present
            return:
                None, error if key is not found
                condition, None if a key is found with expected value
                None, None if every key does not match the expected values
        """
    for condition, (key, value) in self.resource_configuration[name]['conditions'].items():
        status = self.get_key_value(results, key)
        if status is None and name == 'snapmirror_relationship' and results and (condition == 'transfer_state'):
            status = 'idle'
        self.states.append(str(status))
        if status == str(value):
            return (condition, None)
        if status is None:
            return (None, 'Cannot find element with name: %s in results: %s' % (key, results if self.use_rest else results.to_string()))
    return (None, None)