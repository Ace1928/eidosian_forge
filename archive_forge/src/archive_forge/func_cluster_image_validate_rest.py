from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def cluster_image_validate_rest(self):
    api = 'cluster/software'
    body = {'version': self.parameters['package_version']}
    query = {'validate_only': 'true'}
    dummy, error = rest_generic.patch_async(self.rest_api, api, None, body, query, timeout=0, job_timeout=self.parameters['timeout'])
    if error:
        return 'Error validating software: %s' % error
    validation_results = None
    for __ in range(30):
        time.sleep(10)
        validation_results = self.cluster_image_get_rest('validation_results')
        if validation_results is not None:
            break
    return validation_results