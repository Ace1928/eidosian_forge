from __future__ import absolute_import, division, print_function
import os
import re
import time
import uuid
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def create_single_device(module, packet_conn, hostname):
    for param in ('hostnames', 'operating_system', 'plan'):
        if not module.params.get(param):
            raise Exception('%s parameter is required for new device.' % param)
    project_id = module.params.get('project_id')
    plan = module.params.get('plan')
    tags = module.params.get('tags')
    user_data = module.params.get('user_data')
    facility = module.params.get('facility')
    operating_system = module.params.get('operating_system')
    locked = module.params.get('locked')
    ipxe_script_url = module.params.get('ipxe_script_url')
    always_pxe = module.params.get('always_pxe')
    if operating_system != 'custom_ipxe':
        for param in ('ipxe_script_url', 'always_pxe'):
            if module.params.get(param):
                raise Exception('%s parameter is not valid for non custom_ipxe operating_system.' % param)
    device = packet_conn.create_device(project_id=project_id, hostname=hostname, tags=tags, plan=plan, facility=facility, operating_system=operating_system, userdata=user_data, locked=locked, ipxe_script_url=ipxe_script_url, always_pxe=always_pxe)
    return device