from __future__ import absolute_import, division, print_function
import json
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.urls import fetch_url
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def associate_floating_ips(module, rest):
    floating_ip = get_floating_ip_details(module, rest)
    droplet = floating_ip['droplet']
    if droplet is not None and str(droplet['id']) in [module.params['droplet_id']]:
        module.exit_json(changed=False)
    else:
        assign_floating_id_to_droplet(module, rest)