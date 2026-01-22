from __future__ import absolute_import, division, print_function
import json
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.urls import fetch_url
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def get_floating_ip_details(module, rest):
    ip = module.params['ip']
    response = rest.get('floating_ips/{0}'.format(ip))
    status_code = response.status_code
    json_data = response.json
    if status_code == 200:
        return json_data['floating_ip']
    else:
        module.fail_json(msg='Error assigning floating ip [{0}: {1}]'.format(status_code, json_data['message']), region=module.params['region'])