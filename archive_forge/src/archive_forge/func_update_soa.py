from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import navigate_hash, GcpSession, GcpModule, GcpRequest, replace_resource_dict
import json
import copy
import datetime
import time
def update_soa(module):
    original_soa = prefetch_soa_resource(module)
    updated_soa = copy.deepcopy(original_soa)
    soa_parts = updated_soa['rrdatas'][0].split(' ')
    soa_parts[2] = str(int(soa_parts[2]) + 1)
    updated_soa['rrdatas'][0] = ' '.join(soa_parts)
    return [original_soa, updated_soa]