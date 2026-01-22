from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
import time
def destroy_domain(self):
    resp = self.delete('domains/%s' % self.domain_name)
    status, json = self.jsonify(resp)
    if status == 204:
        return True
    else:
        return json