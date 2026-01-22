from __future__ import absolute_import, division, print_function
import json
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.urls import fetch_url
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def assign_floating_id_to_droplet(module, rest):
    ip = module.params['ip']
    payload = {'type': 'assign', 'droplet_id': module.params['droplet_id']}
    response = rest.post('floating_ips/{0}/actions'.format(ip), data=payload)
    status_code = response.status_code
    json_data = response.json
    if status_code == 201:
        json_data = wait_action(module, rest, ip, json_data['action']['id'])
        module.exit_json(changed=True, data=json_data)
    else:
        module.fail_json(msg='Error creating floating ip [{0}: {1}]'.format(status_code, json_data['message']), region=module.params['region'])