from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
def get_service_id_by_name(module, auth, service_name):
    service = get_service_by_name(module, auth, service_name)
    if service:
        return service['ID']
    return None