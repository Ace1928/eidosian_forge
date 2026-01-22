from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def cluster_image_update_rest(self):
    api = 'cluster/software'
    body = {'version': self.parameters['package_version']}
    query = {}
    params_to_rest = {'ignore_validation_warning': 'skip_warnings', 'nodes': 'nodes_to_update', 'stabilize_minutes': 'stabilize_minutes'}
    for param_key, rest_key in params_to_rest.items():
        value = self.parameters.get(param_key)
        if value is not None:
            query[rest_key] = ','.join(value) if rest_key == 'nodes_to_update' else value
    dummy, error = rest_generic.patch_async(self.rest_api, api, None, body, query=query or None, timeout=0, job_timeout=self.parameters['timeout'])
    if self.error_is_fatal(error):
        validation_results, v_error = self.cluster_image_get_rest('validation_results', fail_on_error=False)
        self.module.fail_json(msg='Error updating software: %s - validation results: %s' % (error, v_error or validation_results))
    return self.wait_for_condition(self.is_update_complete_rest, 'image update state')