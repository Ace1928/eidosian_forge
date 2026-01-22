from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def detach_block_storage(self):
    volume_name = self.get_key_or_fail('volume_name')
    region = self.get_key_or_fail('region')
    droplet_id = self.get_key_or_fail('droplet_id')
    changed_status = self.attach_detach_block_storage('detach', volume_name, region, droplet_id)
    self.module.exit_json(changed=changed_status)