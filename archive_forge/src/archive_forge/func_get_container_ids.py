from __future__ import absolute_import, division, print_function
import datetime
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.urls import open_url
def get_container_ids(auth_content, containers):
    host_ids = []
    lower_containers = [x.lower() for x in containers]
    for result in auth_content['result']:
        if result['containers'][0]['name'].lower() in lower_containers:
            data = {'component_id': result['_id'], 'container_id': result['containers'][0]['_id']}
            host_ids.append(data)
            lower_containers.remove(result['containers'][0]['name'].lower())
    if len(lower_containers):
        return (1, None, lower_containers)
    return (0, host_ids, None)