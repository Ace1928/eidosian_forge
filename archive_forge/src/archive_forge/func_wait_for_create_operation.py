from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import navigate_hash, GcpSession, GcpModule, GcpRequest, replace_resource_dict
import json
import time
def wait_for_create_operation(module, response):
    op_result = return_if_object(module, response, 'sql#operation')
    if op_result is None:
        return {}
    status = navigate_hash(op_result, ['operation', 'status'])
    wait_done = wait_for_create_completion(status, op_result, module)
    res = navigate_hash(op_result, ['clientCert', 'certInfo'])
    res.update({'privateKey': navigate_hash(op_result, ['clientCert', 'certPrivateKey'])})
    return res