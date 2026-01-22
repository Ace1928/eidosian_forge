from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import navigate_hash, GcpSession, GcpModule, GcpRequest, replace_resource_dict
import json
import copy
import datetime
import time
def resource_to_change_request(original_record, updated_record, module):
    original_soa, updated_soa = update_soa(module)
    result = new_change_request()
    add_additions(result, updated_soa, updated_record)
    add_deletions(result, original_soa, original_record)
    return result