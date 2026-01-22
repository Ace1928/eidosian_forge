from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def ensure_power_off(self, droplet_id):
    self.wait_status(droplet_id, ['active'])
    self.wait_action(droplet_id, {'type': 'power_off'})