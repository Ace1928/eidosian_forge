from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import navigate_hash, GcpSession, GcpModule, GcpRequest, replace_resource_dict
import json
import copy
import datetime
import time
def get_change_status(change_id, module):
    auth = GcpSession(module, 'dns')
    link = collection(module) + '/%s' % change_id
    return return_if_change_object(module, auth.get(link))['status']