from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def get_kubernetes(self):
    """Returns an existing DigitalOcean Kubernetes cluster by name"""
    json_data = self.get_by_name(self.module.params['name'])
    if json_data:
        self.cluster_id = json_data['id']
        return json_data
    else:
        return None