from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def get_all_clusters(self):
    """Returns all DigitalOcean Kubernetes clusters"""
    response = self.rest.get('kubernetes/clusters')
    json_data = response.json
    if response.status_code == 200:
        return json_data
    return None