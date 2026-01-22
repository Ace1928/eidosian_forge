from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
def check_change_service_owner(module, service, owner_id):
    old_owner_id = int(service['UID'])
    return old_owner_id != owner_id