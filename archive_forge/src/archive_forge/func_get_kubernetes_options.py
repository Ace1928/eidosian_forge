from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def get_kubernetes_options(self):
    """Fetches DigitalOcean Kubernetes options: regions, sizes, versions.
        API reference: https://docs.digitalocean.com/reference/api/api-reference/#operation/list_kubernetes_options
        """
    response = self.rest.get('kubernetes/options')
    json_data = response.json
    if response.status_code == 200:
        return json_data
    return None