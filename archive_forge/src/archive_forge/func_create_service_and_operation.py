from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
def create_service_and_operation(module, auth, template_id, service_name, owner_id, group_id, permissions, custom_attrs, unique, wait, wait_timeout):
    if not service_name:
        service_name = ''
    changed = False
    service = None
    if unique:
        service = get_service_by_name(module, auth, service_name)
    if not service:
        if not module.check_mode:
            service = create_service(module, auth, template_id, service_name, custom_attrs, unique, wait, wait_timeout)
        changed = True
    if module.check_mode and changed:
        return {'changed': True}
    result = service_operation(module, auth, owner_id=owner_id, group_id=group_id, wait=wait, wait_timeout=wait_timeout, permissions=permissions, service=service)
    if result['changed']:
        changed = True
    result['changed'] = changed
    return result