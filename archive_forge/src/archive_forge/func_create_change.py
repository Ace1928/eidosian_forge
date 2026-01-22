from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import navigate_hash, GcpSession, GcpModule, GcpRequest, replace_resource_dict
import json
import copy
import datetime
import time
def create_change(original, updated, module):
    auth = GcpSession(module, 'dns')
    return return_if_change_object(module, auth.post(collection(module), resource_to_change_request(original, updated, module)))