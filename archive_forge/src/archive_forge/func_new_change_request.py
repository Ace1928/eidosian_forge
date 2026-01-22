from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import navigate_hash, GcpSession, GcpModule, GcpRequest, replace_resource_dict
import json
import copy
import datetime
import time
def new_change_request():
    return {'kind': 'dns#change', 'additions': [], 'deletions': [], 'start_time': datetime.datetime.now().isoformat()}