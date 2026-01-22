from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_record_rest(self, name, rest_api_kwargs):
    record, error = rest_generic.get_one_record(self.rest_api, **rest_api_kwargs)
    if error:
        return (None, 'Error running command %s: %s' % (self.parameters['name'], error))
    if not record:
        return (None, 'no record for node: %s' % rest_api_kwargs['query'])
    return (record, None)