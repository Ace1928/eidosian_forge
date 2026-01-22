from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def get_cdn_endpoints(self):
    cdns = self.rest.get_paginated_data(base_url='cdn/endpoints?', data_key_name='endpoints')
    return cdns